//! Pure-Rust `EXT_meshopt_compression` decoder for tile content (BEVY-3D-TILES
//! T6, decision D12 — we EMIT meshopt, never Draco, and decode it on the CPU).
//!
//! A faithful port of the reference decoder bundled with the writer's encoder
//! (`meshoptimizer/meshopt_decoder_reference.js`, by Jasper St. Pierre / Arseny
//! Kapoulkine, MIT) — chosen over the `meshopt` FFI crate so the wasm build
//! stays C-toolchain-free (the plan's "(T6) meshopt C FFI complicates the wasm
//! build" risk). The codec decodes at GB/s and is small and self-contained, so
//! the port is ~one screen per stage:
//!
//! * [`decode_vertex_buffer`] — `ATTRIBUTES` mode (v0 `0xa0` + v1 `0xa1`),
//! * [`decode_index_buffer`] — `TRIANGLES` mode (`0xe1`),
//! * [`decode_index_sequence`] — `INDICES` mode (`0xd1`),
//! * filters `NONE` / `OCTAHEDRAL` / `QUATERNION` / `EXPONENTIAL` (+ `COLOR`).
//!
//! The codec is **byte-lossless**: with filter `NONE` (what our writer emits —
//! `EXTMeshoptCompression` under the QUANTIZE method with no prior
//! quantization), decoded bytes equal the uncompressed source. The triangle
//! codec preserves triangle order and winding, cyclically rotating each
//! triangle (a rendering no-op) — so positions/attributes are byte-identical
//! and the rendered mesh is identical.
//!
//! Filters are lossy by construction (they exist for external/FILTER-method
//! tilesets); our own tiles never carry one. `js_round` mirrors JavaScript's
//! `Math.round` (round half toward +∞) so filter output is byte-for-byte equal
//! to the reference decoder.

/// glTF "zig-zag"-style decode: even → `v/2`, odd → bitwise-NOT of `v/2`
/// (the wrapped negative). Callers keep only the low bits they need.
#[inline]
fn dezig(v: u32) -> u32 {
    if v & 1 != 0 { !(v >> 1) } else { v >> 1 }
}

/// JavaScript `Math.round`: round half toward +∞ (`floor(x + 0.5)`), which the
/// reference filters use — `f64::round` (half away from zero) disagrees on
/// negative halves and would desync byte-for-byte.
#[inline]
fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// Decode one EXT_meshopt_compression buffer view: run the `mode` codec into a
/// fresh `count * byte_stride` byte buffer, then apply `filter`. `filter` is
/// the extension's string (`""`/absent ⇒ `NONE`).
pub fn decode_buffer_view(
    mode: &str,
    filter: &str,
    count: usize,
    byte_stride: usize,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; count * byte_stride];
    match mode {
        "ATTRIBUTES" => decode_vertex_buffer(&mut out, count, byte_stride, source)?,
        "TRIANGLES" => decode_index_buffer(&mut out, count, byte_stride, source)?,
        "INDICES" => decode_index_sequence(&mut out, count, byte_stride, source)?,
        other => return Err(format!("meshopt: unknown mode {other}")),
    }
    match filter {
        "" | "NONE" => {}
        "OCTAHEDRAL" => filter_octahedral(&mut out, count, byte_stride)?,
        "QUATERNION" => filter_quaternion(&mut out, count, byte_stride)?,
        "EXPONENTIAL" => filter_exponential(&mut out, count, byte_stride)?,
        "COLOR" => filter_color(&mut out, count, byte_stride)?,
        other => return Err(format!("meshopt: unknown filter {other}")),
    }
    Ok(out)
}

// ── Vertex buffer (ATTRIBUTES) ───────────────────────────────────────────────

/// Decode a meshopt vertex buffer into `target` (`element_count *
/// byte_stride`). Supports format v0 (`0xa0`) and v1 (`0xa1`).
pub fn decode_vertex_buffer(
    target: &mut [u8],
    element_count: usize,
    byte_stride: usize,
    source: &[u8],
) -> Result<(), String> {
    if byte_stride == 0 || !byte_stride.is_multiple_of(4) {
        return Err(format!("meshopt vertex: byte_stride {byte_stride} not a multiple of 4"));
    }
    let header = *source.first().ok_or("meshopt vertex: empty source")?;
    if header != 0xa0 && header != 0xa1 {
        return Err(format!("meshopt vertex: bad header byte 0x{header:02x}"));
    }
    let version = (header & 0x0f) as usize; // 0 or 1
    if target.len() != element_count * byte_stride {
        return Err("meshopt vertex: target size mismatch".into());
    }

    let max_block_elements = ((0x2000 / byte_stride) & !0x0f).clamp(1, 0x100);
    // +16: the reference relies on out-of-bounds TypedArray writes being no-ops
    // for the final byte's tail group; a padded buffer makes those writes land
    // harmlessly (they are never read back).
    let mut deltas = vec![0u8; max_block_elements * byte_stride + 16];

    let tail_size = if version == 0 { byte_stride } else { byte_stride + byte_stride / 4 };
    if source.len() < 1 + tail_size {
        return Err("meshopt vertex: source too short for tail".into());
    }
    let tail_data_offs = source.len() - tail_size;
    // `temp_data` is the running "previous element"; the tail seeds it.
    let mut temp_data = source[tail_data_offs..tail_data_offs + byte_stride].to_vec();
    // v1 per-4-byte channel-mode bytes (channel mode + rotation), else absent.
    let channels: Vec<u8> = if version == 0 {
        Vec::new()
    } else {
        source[tail_data_offs + byte_stride..tail_data_offs + tail_size].to_vec()
    };

    // Header-mode tables: [v0], [v1 control 0], [v1 control 1].
    const HEADER_MODES: [[u32; 4]; 3] = [[0, 2, 4, 8], [0, 1, 2, 4], [1, 2, 4, 8]];

    let mut src_offs = 1usize; // skip header byte

    let mut dst_elem_base = 0usize;
    while dst_elem_base < element_count {
        let abc = (element_count - dst_elem_base).min(max_block_elements);
        let group_count = ((abc + 0x0f) & !0x0f) >> 4;
        let header_byte_count = ((group_count + 0x03) & !0x03) >> 2;

        let control_bits_offs = src_offs;
        src_offs += if version == 0 { 0 } else { byte_stride / 4 };

        for d in deltas.iter_mut() {
            *d = 0;
        }

        for byte in 0..byte_stride {
            let delta_base = byte * abc;

            let control_mode = if version == 0 {
                0u32
            } else {
                ((source[control_bits_offs + (byte >> 2)] >> ((byte & 0x03) << 1)) & 0x03) as u32
            };

            if control_mode == 2 {
                continue; // all deltas 0
            } else if control_mode == 3 {
                // Stored uncompressed, no header bits.
                deltas[delta_base..delta_base + abc]
                    .copy_from_slice(&source[src_offs..src_offs + abc]);
                src_offs += abc;
                continue;
            }

            let header_bits_offs = src_offs;
            src_offs += header_byte_count;

            for group in 0..group_count {
                let mode = ((source[header_bits_offs + (group >> 2)] >> ((group & 0x03) << 1))
                    & 0x03) as usize;
                let table = if version == 0 { 0 } else { (control_mode + 1) as usize };
                let mode_bits = HEADER_MODES[table][mode];
                let delta_offs = delta_base + (group << 4);

                match mode_bits {
                    0 => {} // all 16 deltas zero
                    1 => {
                        let src_base = src_offs;
                        src_offs += 2;
                        for m in 0..16usize {
                            let shift = m & 0x07;
                            let mut delta = (source[src_base + (m >> 3)] >> shift) & 0x01;
                            if delta == 1 {
                                delta = source[src_offs];
                                src_offs += 1;
                            }
                            deltas[delta_offs + m] = delta;
                        }
                    }
                    2 => {
                        let src_base = src_offs;
                        src_offs += 4;
                        for m in 0..16usize {
                            let shift = 6 - ((m & 0x03) << 1);
                            let mut delta = (source[src_base + (m >> 2)] >> shift) & 0x03;
                            if delta == 3 {
                                delta = source[src_offs];
                                src_offs += 1;
                            }
                            deltas[delta_offs + m] = delta;
                        }
                    }
                    4 => {
                        let src_base = src_offs;
                        src_offs += 8;
                        for m in 0..16usize {
                            let shift = 4 - ((m & 0x01) << 2);
                            let mut delta = (source[src_base + (m >> 1)] >> shift) & 0x0f;
                            if delta == 0x0f {
                                delta = source[src_offs];
                                src_offs += 1;
                            }
                            deltas[delta_offs + m] = delta;
                        }
                    }
                    _ => {
                        // 8: 16 verbatim bytes.
                        deltas[delta_offs..delta_offs + 16]
                            .copy_from_slice(&source[src_offs..src_offs + 16]);
                        src_offs += 16;
                    }
                }
            }
        }

        // Apply deltas per element, per 4-byte channel group.
        for elem in 0..abc {
            let dst_elem = dst_elem_base + elem;
            let mut byte_group = 0usize;
            while byte_group < byte_stride {
                let channel_mode =
                    if version == 0 { 0u32 } else { (channels[byte_group >> 2] & 0x03) as u32 };
                if channel_mode == 3 {
                    return Err("meshopt vertex: reserved channel mode 3".into());
                }

                if channel_mode == 0 {
                    for byte in byte_group..byte_group + 4 {
                        let delta = dezig(deltas[byte * abc + elem] as u32);
                        let temp = (temp_data[byte] as u32).wrapping_add(delta) & 0xff;
                        target[dst_elem * byte_stride + byte] = temp as u8;
                        temp_data[byte] = temp as u8;
                    }
                } else if channel_mode == 1 {
                    let mut byte = byte_group;
                    while byte < byte_group + 4 {
                        let d = (deltas[byte * abc + elem] as u32)
                            | ((deltas[(byte + 1) * abc + elem] as u32) << 8);
                        let delta = dezig(d);
                        let mut temp =
                            (temp_data[byte] as u32) | ((temp_data[byte + 1] as u32) << 8);
                        temp = temp.wrapping_add(delta) & 0xffff;
                        let dst = dst_elem * byte_stride + byte;
                        target[dst] = temp as u8;
                        temp_data[byte] = temp as u8;
                        target[dst + 1] = (temp >> 8) as u8;
                        temp_data[byte + 1] = (temp >> 8) as u8;
                        byte += 2;
                    }
                } else {
                    // channel_mode == 2: 4-byte rotate-XOR delta.
                    let byte = byte_group;
                    let delta = (deltas[byte * abc + elem] as u32)
                        | ((deltas[(byte + 1) * abc + elem] as u32) << 8)
                        | ((deltas[(byte + 2) * abc + elem] as u32) << 16)
                        | ((deltas[(byte + 3) * abc + elem] as u32) << 24);
                    let mut temp = (temp_data[byte] as u32)
                        | ((temp_data[byte + 1] as u32) << 8)
                        | ((temp_data[byte + 2] as u32) << 16)
                        | ((temp_data[byte + 3] as u32) << 24);
                    let rot = (channels[byte_group >> 2] >> 4) as u32;
                    temp ^= delta.rotate_right(rot);
                    let dst = dst_elem * byte_stride + byte;
                    for k in 0..4 {
                        let v = (temp >> (k * 8)) as u8;
                        target[dst + k] = v;
                        temp_data[byte + k] = v;
                    }
                }
                byte_group += 4;
            }
        }

        dst_elem_base += max_block_elements;
    }

    let tail_size_padded = tail_size.max(if version == 0 { 32 } else { 24 });
    if src_offs != source.len() - tail_size_padded {
        return Err(format!(
            "meshopt vertex: consumed {src_offs} bytes, expected {}",
            source.len() - tail_size_padded
        ));
    }
    Ok(())
}

// ── Index buffer (TRIANGLES) ─────────────────────────────────────────────────

#[inline]
fn pushfifo(fifo: &mut [u32], n: u32) {
    for i in (1..fifo.len()).rev() {
        fifo[i] = fifo[i - 1];
    }
    fifo[0] = n;
}

#[inline]
fn read_leb128(source: &[u8], data_offs: &mut usize) -> u32 {
    let mut n: u32 = 0;
    let mut i: u32 = 0;
    loop {
        let b = source[*data_offs] as u32;
        *data_offs += 1;
        n |= (b & 0x7f) << i;
        if b < 0x80 {
            return n;
        }
        i += 7;
    }
}

#[inline]
fn write_index(target: &mut [u8], byte_stride: usize, pos: usize, value: u32) {
    let off = pos * byte_stride;
    if byte_stride == 2 {
        target[off..off + 2].copy_from_slice(&(value as u16).to_le_bytes());
    } else {
        target[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// Decode a meshopt triangle index buffer (`count` indices, `count % 3 == 0`).
pub fn decode_index_buffer(
    target: &mut [u8],
    count: usize,
    byte_stride: usize,
    source: &[u8],
) -> Result<(), String> {
    if source.first() != Some(&0xe1) {
        return Err("meshopt index: bad header (expected 0xe1)".into());
    }
    if !count.is_multiple_of(3) {
        return Err("meshopt index: count not a multiple of 3".into());
    }
    if byte_stride != 2 && byte_stride != 4 {
        return Err("meshopt index: byte_stride must be 2 or 4".into());
    }
    if source.len() < 16 {
        return Err("meshopt index: source too short".into());
    }

    let tri_count = count / 3;
    // Codes are a flat array right after the header (one byte per triangle);
    // `data_offs` (LEB128 stream) and `codeaux_offs` advance independently, so
    // the triangle index doubles as the code cursor (`1 + tri`).
    let mut data_offs = 1 + tri_count;
    let codeaux_offs = source.len() - 16;

    let mut next: u32 = 0;
    let mut last: u32 = 0;
    let mut edgefifo = [0u32; 32];
    let mut vertexfifo = [0u32; 16];

    let mut dst_offs = 0usize;
    for tri in 0..tri_count {
        let code = source[1 + tri];
        let b0 = (code >> 4) as usize;
        let b1 = (code & 0x0f) as usize;

        if b0 < 0x0f {
            // Reuse an existing edge (a, b) from the edge fifo; emit (a, b, c).
            let ea = edgefifo[b0 << 1];
            let eb = edgefifo[(b0 << 1) + 1];
            let cc = match b1 {
                0x00 => {
                    let v = next;
                    next = next.wrapping_add(1);
                    pushfifo(&mut vertexfifo, v);
                    v
                }
                0x0d => {
                    last = last.wrapping_sub(1);
                    pushfifo(&mut vertexfifo, last);
                    last
                }
                0x0e => {
                    last = last.wrapping_add(1);
                    pushfifo(&mut vertexfifo, last);
                    last
                }
                0x0f => {
                    let v = read_leb128(source, &mut data_offs);
                    last = last.wrapping_add(dezig(v));
                    pushfifo(&mut vertexfifo, last);
                    last
                }
                _ => vertexfifo[b1], // 0x01..=0x0c
            };
            // fifo pushes happen "backwards" (see reference).
            pushfifo(&mut edgefifo, eb);
            pushfifo(&mut edgefifo, cc);
            pushfifo(&mut edgefifo, cc);
            pushfifo(&mut edgefifo, ea);
            write_index(target, byte_stride, dst_offs, ea);
            write_index(target, byte_stride, dst_offs + 1, eb);
            write_index(target, byte_stride, dst_offs + 2, cc);
            dst_offs += 3;
        } else {
            // b0 == 0x0f
            let (aa, bb, cc);
            if b1 < 0x0e {
                let e = source[codeaux_offs + b1];
                let z = (e >> 4) as usize;
                let w = (e & 0x0f) as usize;
                aa = next;
                next = next.wrapping_add(1);
                bb = if z == 0 {
                    let v = next;
                    next = next.wrapping_add(1);
                    v
                } else {
                    vertexfifo[z - 1]
                };
                cc = if w == 0 {
                    let v = next;
                    next = next.wrapping_add(1);
                    v
                } else {
                    vertexfifo[w - 1]
                };
                pushfifo(&mut vertexfifo, aa);
                if z == 0 {
                    pushfifo(&mut vertexfifo, bb);
                }
                if w == 0 {
                    pushfifo(&mut vertexfifo, cc);
                }
            } else {
                let e = source[data_offs];
                data_offs += 1;
                if e == 0 {
                    next = 0;
                }
                let z = (e >> 4) as usize;
                let w = (e & 0x0f) as usize;
                aa = if b1 == 0x0e {
                    let v = next;
                    next = next.wrapping_add(1);
                    v
                } else {
                    let v = read_leb128(source, &mut data_offs);
                    last = last.wrapping_add(dezig(v));
                    last
                };
                bb = if z == 0 {
                    let v = next;
                    next = next.wrapping_add(1);
                    v
                } else if z == 0x0f {
                    let v = read_leb128(source, &mut data_offs);
                    last = last.wrapping_add(dezig(v));
                    last
                } else {
                    vertexfifo[z - 1]
                };
                cc = if w == 0 {
                    let v = next;
                    next = next.wrapping_add(1);
                    v
                } else if w == 0x0f {
                    let v = read_leb128(source, &mut data_offs);
                    last = last.wrapping_add(dezig(v));
                    last
                } else {
                    vertexfifo[w - 1]
                };
                pushfifo(&mut vertexfifo, aa);
                if z == 0 || z == 0x0f {
                    pushfifo(&mut vertexfifo, bb);
                }
                if w == 0 || w == 0x0f {
                    pushfifo(&mut vertexfifo, cc);
                }
            }
            pushfifo(&mut edgefifo, aa);
            pushfifo(&mut edgefifo, bb);
            pushfifo(&mut edgefifo, bb);
            pushfifo(&mut edgefifo, cc);
            pushfifo(&mut edgefifo, cc);
            pushfifo(&mut edgefifo, aa);
            write_index(target, byte_stride, dst_offs, aa);
            write_index(target, byte_stride, dst_offs + 1, bb);
            write_index(target, byte_stride, dst_offs + 2, cc);
            dst_offs += 3;
        }
    }
    Ok(())
}

// ── Index sequence (INDICES) ─────────────────────────────────────────────────

/// Decode a meshopt index sequence (`count` indices, any topology).
pub fn decode_index_sequence(
    target: &mut [u8],
    count: usize,
    byte_stride: usize,
    source: &[u8],
) -> Result<(), String> {
    if source.first() != Some(&0xd1) {
        return Err("meshopt sequence: bad header (expected 0xd1)".into());
    }
    if byte_stride != 2 && byte_stride != 4 {
        return Err("meshopt sequence: byte_stride must be 2 or 4".into());
    }
    let mut data_offs = 1usize;
    let mut last = [0u32; 2];
    for i in 0..count {
        let v = read_leb128(source, &mut data_offs);
        let b = (v & 1) as usize;
        let delta = dezig(v >> 1);
        last[b] = last[b].wrapping_add(delta);
        write_index(target, byte_stride, i, last[b]);
    }
    Ok(())
}

// ── Filters ──────────────────────────────────────────────────────────────────

#[inline]
fn read_i16(buf: &[u8], i: usize) -> i32 {
    i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]) as i32
}
#[inline]
fn write_i16(buf: &mut [u8], i: usize, v: i32) {
    buf[i * 2..i * 2 + 2].copy_from_slice(&(v as i16).to_le_bytes());
}

/// Octahedral normal/tangent decode (`byte_stride` 4 ⇒ i8, 8 ⇒ i16).
fn filter_octahedral(target: &mut [u8], count: usize, byte_stride: usize) -> Result<(), String> {
    match byte_stride {
        4 => {
            let max_int = 127.0f64;
            for e in 0..count {
                let base = e * 4;
                let mut x = (target[base] as i8) as f64;
                let mut y = (target[base + 1] as i8) as f64;
                let one = (target[base + 2] as i8) as f64;
                x /= one;
                y /= one;
                let z = 1.0 - x.abs() - y.abs();
                let t = (-z).max(0.0);
                x -= if x >= 0.0 { t } else { -t };
                y -= if y >= 0.0 { t } else { -t };
                let h = max_int / (x * x + y * y + z * z).sqrt();
                target[base] = (js_round(x * h) as i32 as i8) as u8;
                target[base + 1] = (js_round(y * h) as i32 as i8) as u8;
                target[base + 2] = (js_round(z * h) as i32 as i8) as u8;
                // target[base + 3] (w / sign) untouched.
            }
        }
        8 => {
            let max_int = 32767.0f64;
            for e in 0..count {
                let base = e * 4; // i16 index base
                let mut x = read_i16(target, base) as f64;
                let mut y = read_i16(target, base + 1) as f64;
                let one = read_i16(target, base + 2) as f64;
                x /= one;
                y /= one;
                let z = 1.0 - x.abs() - y.abs();
                let t = (-z).max(0.0);
                x -= if x >= 0.0 { t } else { -t };
                y -= if y >= 0.0 { t } else { -t };
                let h = max_int / (x * x + y * y + z * z).sqrt();
                write_i16(target, base, js_round(x * h) as i32);
                write_i16(target, base + 1, js_round(y * h) as i32);
                write_i16(target, base + 2, js_round(z * h) as i32);
            }
        }
        other => return Err(format!("meshopt OCTAHEDRAL: byte_stride {other} (need 4 or 8)")),
    }
    Ok(())
}

/// Quaternion decode (`byte_stride` 8, four i16). The largest component is
/// reconstructed from the other three; its slot is carried in the low 2 bits.
fn filter_quaternion(target: &mut [u8], count: usize, byte_stride: usize) -> Result<(), String> {
    if byte_stride != 8 {
        return Err(format!("meshopt QUATERNION: byte_stride {byte_stride} (need 8)"));
    }
    const SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    for e in 0..count {
        let base = e * 4; // i16 index base
        let input_w = read_i16(target, base + 3);
        let max_component = (input_w & 0x03) as usize;
        let s = SQRT1_2 / ((input_w | 0x03) as f64);
        let x = read_i16(target, base) as f64 * s;
        let y = read_i16(target, base + 1) as f64 * s;
        let z = read_i16(target, base + 2) as f64 * s;
        let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
        write_i16(target, base + (max_component + 1) % 4, js_round(x * 32767.0) as i32);
        write_i16(target, base + (max_component + 2) % 4, js_round(y * 32767.0) as i32);
        write_i16(target, base + (max_component + 3) % 4, js_round(z * 32767.0) as i32);
        write_i16(target, base + max_component % 4, js_round(w * 32767.0) as i32);
    }
    Ok(())
}

/// Exponential decode: each 4 bytes is `int8 exponent << 24 | int24 mantissa`
/// → `2^exp * mantissa` as f32.
fn filter_exponential(target: &mut [u8], count: usize, byte_stride: usize) -> Result<(), String> {
    if !byte_stride.is_multiple_of(4) {
        return Err(format!("meshopt EXPONENTIAL: byte_stride {byte_stride} not /4"));
    }
    let n = byte_stride * count / 4;
    for i in 0..n {
        let v = i32::from_le_bytes([
            target[i * 4],
            target[i * 4 + 1],
            target[i * 4 + 2],
            target[i * 4 + 3],
        ]);
        let exp = v >> 24;
        let mantissa = (v << 8) >> 8;
        let f = 2.0f64.powi(exp) * (mantissa as f64);
        target[i * 4..i * 4 + 4].copy_from_slice(&(f as f32).to_le_bytes());
    }
    Ok(())
}

/// YCoCg-R color decode (`byte_stride` 4 ⇒ u8, 8 ⇒ u16) with scale carried in
/// the alpha high bit. Ported for completeness (newer meshopt assets).
fn filter_color(target: &mut [u8], count: usize, byte_stride: usize) -> Result<(), String> {
    let max_int = ((1u64 << (byte_stride * 2)) - 1) as f64;
    match byte_stride {
        4 => {
            for e in 0..count {
                let base = e * 4;
                let y = target[base] as f64;
                let co = (target[base + 1] as i8) as f64;
                let cg = (target[base + 2] as i8) as f64;
                let alpha_input = target[base + 3] as u32;
                let alpha_bit = if alpha_input == 0 { -1i32 } else { 31 - alpha_input.leading_zeros() as i32 };
                let as_ = ((1i64 << (alpha_bit + 1)) - 1) as f64;
                let r = y + co - cg;
                let g = y + cg;
                let b = y - co - cg;
                let mut a = (alpha_input as i64) & ((as_ as i64) >> 1);
                a = (a << 1) | (a & 1);
                let ss = max_int / as_;
                target[base] = js_round(r * ss) as i64 as u8;
                target[base + 1] = js_round(g * ss) as i64 as u8;
                target[base + 2] = js_round(b * ss) as i64 as u8;
                target[base + 3] = js_round(a as f64 * ss) as i64 as u8;
            }
        }
        8 => {
            for e in 0..count {
                let base = e * 4; // u16 index base
                let y = u16::from_le_bytes([target[base * 2], target[base * 2 + 1]]) as f64;
                let co = read_i16(target, base + 1) as f64;
                let cg = read_i16(target, base + 2) as f64;
                let alpha_input =
                    u16::from_le_bytes([target[(base + 3) * 2], target[(base + 3) * 2 + 1]]) as u32;
                let alpha_bit = if alpha_input == 0 { -1i32 } else { 31 - alpha_input.leading_zeros() as i32 };
                let as_ = ((1i64 << (alpha_bit + 1)) - 1) as f64;
                let r = y + co - cg;
                let g = y + cg;
                let b = y - co - cg;
                let mut a = (alpha_input as i64) & ((as_ as i64) >> 1);
                a = (a << 1) | (a & 1);
                let ss = max_int / as_;
                let w = |buf: &mut [u8], i: usize, v: f64| {
                    let u = js_round(v) as i64 as u16;
                    buf[i * 2..i * 2 + 2].copy_from_slice(&u.to_le_bytes());
                };
                w(target, base, r * ss);
                w(target, base + 1, g * ss);
                w(target, base + 2, b * ss);
                w(target, base + 3, a as f64 * ss);
            }
        }
        other => return Err(format!("meshopt COLOR: byte_stride {other} (need 4 or 8)")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a compact hex string into bytes (test vectors are captured from
    /// the bundled `meshoptimizer` encoder + reference decoder; see
    /// `gen_meshopt_vectors` notes in the BEVY-3D-TILES T6 commit).
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn u32s_le(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn vertex_buffer_round_trips_byte_identical() {
        // 8 × VEC3 f32 positions, ATTRIBUTES/NONE — the lossless path our
        // writer uses (QUANTIZE method, no quantization).
        let enc = hex("a00000013fcf0000ffffffffffbf013fff00007e7d7e7d80fefb0000010cff0000ffff807fbf010cff00007e7d80fffd00000100cf0000ff7f9f0100ef00007efffd0000000000000000000000000000000000000000000000000000000000000000");
        let expect = hex("0000000000000000000000000000803f0000000000000000000000000000803f000000000000803f0000803f0000000000000000000000000000803f000000400000404000008040000080bf000000c0000040c0000020410000a0410000f041");
        let out = decode_buffer_view("ATTRIBUTES", "NONE", 8, 12, &enc).unwrap();
        assert_eq!(out, expect, "vertex codec must be byte-lossless");
    }

    #[test]
    fn index_buffer_preserves_triangles_and_winding() {
        // Source tris [0,1,2][2,1,3][0,2,4][5,6,7][4,2,0]; meshopt rotates the
        // last to [0,4,2] (winding preserved — a rendering no-op).
        let enc = hex("e1f01020f035007687566778a9866589689801690000");
        let expect = u32s_le(&[0, 1, 2, 2, 1, 3, 0, 2, 4, 5, 6, 7, 0, 4, 2]);
        let out = decode_buffer_view("TRIANGLES", "NONE", 15, 4, &enc).unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn index_sequence_round_trips() {
        let enc = hex("d100040404040c020200000000");
        let expect = u32s_le(&[0, 1, 2, 3, 4, 7, 6, 5]);
        let out = decode_buffer_view("INDICES", "NONE", 8, 4, &enc).unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn octahedral_filter_matches_reference() {
        let enc = hex("a0013f000000fefda9010f000000fe5800000000000000000000000000000000000000000000000000000000000000007f00");
        let expect = hex("00007f007f000000007f0000b7b7b600");
        let out = decode_buffer_view("ATTRIBUTES", "OCTAHEDRAL", 4, 4, &enc).unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn quaternion_filter_matches_reference() {
        let enc = hex("a00107000000af010f0000000e030103000000b101030000000a011b000000b1013f0000000e0d0a01390000000500000000000000000000000000000000000000000000000000000000000000ff07");
        let expect = hex("000000000000ff7f825a00000000825a0000825a825a00000f40fa3ffa3ffa3f");
        let out = decode_buffer_view("ATTRIBUTES", "QUATERNION", 4, 8, &enc).unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn exponential_filter_matches_reference() {
        let enc = hex("a0013c0000006e87013c000000640f00013c0000000908013000000080013c00000004540120000000013c0000000908013c0000000708013c00000065560118000000013c000000090800000000000000000000000000000000000000000d0000f9c0fefff9003200f9");
        let expect = hex("0000d03d000020c00000c8420010494000000000000080ba000028420000284200002842");
        let out = decode_buffer_view("ATTRIBUTES", "EXPONENTIAL", 3, 12, &enc).unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn bad_headers_error_cleanly() {
        assert!(decode_buffer_view("ATTRIBUTES", "NONE", 1, 12, &[0x00; 40]).is_err());
        assert!(decode_buffer_view("TRIANGLES", "NONE", 3, 4, &[0x00; 20]).is_err());
        assert!(decode_buffer_view("WHAT", "NONE", 1, 4, &[0xa0]).is_err());
    }
}
