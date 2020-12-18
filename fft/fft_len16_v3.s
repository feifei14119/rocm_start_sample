/*
 * 0. modify hsa_code_object_isa
 * 1. modify kernel name
 * 2. modify GROUP_NUM or GROUP_SIZE
 * 3. modify kernel arguments offset in constanand define and amd_amdgpu_hsa_metadata, and args_size value
 * 4. modify sgpr allocate and sgpr_count value
 * 5. modify vgpr allocate and vgpr_count value
 * 6. modify lds allocate and lds_byte_size value
*/

.text
.globl fft_len16
.p2align 6
.type fft_len16,@function

/* constanand define */
.set WAVE_SIZE,				(64)
.set SIMD_NUM,				(1)
.set GROUP_NUM,				(1)
.set GROUP_SIZE, 			(WAVE_SIZE * SIMD_NUM)
.set GLOBAL_SIZE,			(GROUP_SIZE * GROUP_NUM)
.set ELEMT_SIZE,			(8) // float2

/* kernel argument offset */
.set arg_x_offset,			(0)
.set arg_w_offset,			(8)
.set arg_y_offset,			(16)
.set arg_len_offset,		(24)

.set args_size,				(28)

/* sgpr allocate */
.set s_addr_args,			(0)
.set s_bid_x,				(2)
.set s_addr_x,				(6)
.set s_addr_w,				(8)
.set s_addr_y,				(10)

.set sgpr_count,			(11 + 1)

/* vgpr allocate */
.set v_tid_x,				(0)
.set v_addr_x,				(2)
.set v_addr_w,				(4)
.set v_addr_y,				(6)

.set v_r0,					(8)
.set v_r1,					(10)
.set v_r2,					(12)
.set v_r3,					(14)
.set v_r0x,					(v_r0)
.set v_r1x,					(v_r1)
.set v_r2x,					(v_r2)
.set v_r3x,					(v_r3)
.set v_r0y,					(v_r0+1)
.set v_r1y,					(v_r1+1)
.set v_r2y,					(v_r2+1)
.set v_r3y,					(v_r3+1)
.set v_w1,					(16)
.set v_w2,					(18)
.set v_w3,					(20)
.set v_w1x,					(v_w1)
.set v_w2x,					(v_w2)
.set v_w3x,					(v_w3)
.set v_w1y,					(v_w1+1)
.set v_w2y,					(v_w2+1)
.set v_w3y,					(v_w3+1)

.set v_tmp_r,               (28)
.set v_tmp_rx,              (28)
.set v_tmp_ry,              (29)
.set v_temp1,				(30)
.set v_temp2,				(31)

.set vgpr_count,			(32)

/* lds allocate */
.set lds_byte_size,			(0)

/* function define */
.macro Log2Func num_lg2, num
	\num_lg2 = 0
	lg_i = \num
	.rept 32
		lg_i = lg_i / 2
		.if lg_i > 0
			\num_lg2 = \num_lg2 + 1
		.endif
	.endr
.endm

.macro FwdRad4B1 r0, r2, r1, r3
    /* (*R1) = (*R0) - (*R1); */
    v_sub_f32			v[\r1],					v[\r0],   v[\r1]
    v_sub_f32			v[\r1+1],				v[\r0+1], v[\r1+1]
    /* (*R0) = 2.0 * (*R0) - (*R1); */
    v_mul_f32_e64		v[v_tmp_r],				v[\r0],   2.0
    v_mul_f32_e64		v[v_tmp_r+1],			v[\r0+1], 2.0
    v_sub_f32			v[\r0],					v[v_tmp_r],   v[\r1]
    v_sub_f32			v[\r0+1],				v[v_tmp_r+1], v[\r1+1]
    /* (*R3) = (*R2) - (*R3); */
    v_sub_f32			v[\r3],					v[\r2],   v[\r3]
    v_sub_f32			v[\r3+1],				v[\r2+1], v[\r3+1]
    /* (*R2) = 2.0 * (*R2) - (*R3); */
    v_mul_f32_e64		v[v_tmp_r],				v[\r2],   2.0
    v_mul_f32_e64		v[v_tmp_r+1],			v[\r2+1], 2.0
    v_sub_f32			v[\r2],					v[v_tmp_r],   v[\r3]
    v_sub_f32			v[\r2+1],				v[v_tmp_r+1], v[\r3+1]

    /* (*R2) = (*R0) - (*R2); */
    v_sub_f32			v[\r2],					v[\r0],   v[\r2]
    v_sub_f32			v[\r2+1],				v[\r0+1], v[\r2+1]    
    /* (*R0) = 2.0 * (*R0) - (*R2); */
    v_mul_f32_e64		v[v_tmp_r],				v[\r0],   2.0
    v_mul_f32_e64		v[v_tmp_r+1],			v[\r0+1], 2.0
    v_sub_f32			v[\r0],					v[v_tmp_r],   v[\r2]
    v_sub_f32			v[\r0+1],				v[v_tmp_r+1], v[\r2+1]
    
    /* (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x); */
    v_mov_b32			v[v_tmp_r],				v[\r3]
    v_add_f32			v[\r3],					v[\r1],  -v[\r3+1]
    v_add_f32			v[\r3+1],				v[\r1+1], v[v_tmp_r]
    /* (*R1) = 2.0 * (*R1) - (*R3); */
    v_mul_f32_e64		v[v_tmp_r],				v[\r1],   2.0
    v_mul_f32_e64		v[v_tmp_r+1],			v[\r1+1], 2.0
    v_sub_f32			v[\r1],					v[v_tmp_r],   v[\r3]
    v_sub_f32			v[\r1+1],				v[v_tmp_r+1], v[\r3+1]

    /*res = (*R1);
    (*R1) = (*R2);
    (*R2) = res;*/
    v_swap_b32			v[\r1],					v[\r2]
    v_swap_b32			v[\r1+1],				v[\r2+1]
.endm

.macro Transpose r0, r1, r2, r3
    s_mov_b64			exec,					0x6
    v_swap_b32			v[\r0],					v[\r2]
    v_swap_b32			v[\r0+1],				v[\r2+1]
    s_mov_b64			exec,					0xC
    v_swap_b32			v[\r1],					v[\r3]
    v_swap_b32			v[\r1+1],				v[\r3+1]
    s_mov_b64			exec,					0xA
    v_swap_b32			v[\r0],					v[\r1]
    v_swap_b32			v[\r0+1],				v[\r1+1]
    v_swap_b32			v[\r2],					v[\r3]
    v_swap_b32			v[\r2+1],				v[\r3+1]
    s_mov_b64			exec,					0xF
    v_mov_b32_dpp		v[\r1],					v[\r1]			quad_perm:[3,0,1,2]
    v_mov_b32_dpp		v[\r1+1],				v[\r1+1]		quad_perm:[3,0,1,2]
    v_mov_b32_dpp		v[\r2],					v[\r2]			quad_perm:[2,3,0,1]
    v_mov_b32_dpp		v[\r2+1],				v[\r2+1]		quad_perm:[2,3,0,1]
    v_mov_b32_dpp		v[\r3],					v[\r3]			quad_perm:[1,2,3,0]
    v_mov_b32_dpp		v[\r3+1],				v[\r3+1]		quad_perm:[1,2,3,0]
    s_mov_b64			exec,					0xA
    v_swap_b32			v[\r0],					v[\r1]
    v_swap_b32			v[\r0+1],				v[\r1+1]
    v_swap_b32			v[\r2],					v[\r3]
    v_swap_b32			v[\r2+1],				v[\r3+1]
    s_mov_b64			exec,					0xC
    v_swap_b32			v[\r0],					v[\r2]
    v_swap_b32			v[\r0+1],				v[\r2+1]
    s_mov_b64			exec,					0x9
    v_swap_b32			v[\r1],					v[\r3]
    v_swap_b32			v[\r1+1],				v[\r3+1]
    s_mov_b64			exec,					0xF
.endm

.macro UpdateW ro, ri, w
	/* TR = wx * rx - wy * ry; */
    v_mul_f32			v[v_tmp_r],				v[\w],   v[\ri]
    v_fma_f32			v[v_tmp_r],				-v[\w+1], v[\ri+1],   v[v_tmp_r]
	/* TI = wy * rx + wx * ry; */
    v_mul_f32			v[v_tmp_r+1],			v[\w+1], v[\ri]
    v_fma_f32			v[v_tmp_r+1],			v[\w],   v[\ri+1],   v[v_tmp_r+1]

    v_mov_b32			v[\ro],					v[v_tmp_r]
    v_mov_b32			v[\ro+1],				v[v_tmp_r+1]
.endm

/* variable  declare */
GroupSizeShift = 0
ElemtByteShift = 0
Log2Func ElemtByteShift, ELEMT_SIZE
Log2Func GroupSizeShift, GROUP_SIZE

FFT_TILE = 4
Log2Func FftTileShift, FFT_TILE

fft_len16:    
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_x:s_addr_x+1], s[s_addr_args:s_addr_args+1], arg_x_offset
    s_load_dwordx2		s[s_addr_w:s_addr_w+1], s[s_addr_args:s_addr_args+1], arg_w_offset
    s_load_dwordx2		s[s_addr_y:s_addr_y+1], s[s_addr_args:s_addr_args+1], arg_y_offset
    s_waitcnt			lgkmcnt(0)

    /* 4 thread do 1 batch transform */ 
    s_mov_b64			exec,					0xF

	/* calculate input address */
    v_lshlrev_b32		v[v_temp1], 			GroupSizeShift, s[s_bid_x] 					// temp1 = bid_x * group_size
    v_add_lshl_u32		v[v_temp1], 			v[v_temp1], v[v_tid_x], ElemtByteShift		// temp1 = (bid_x * group_size + tid_x) * 4
    v_lshlrev_b32		v[v_temp1], 			FftTileShift, v[v_temp1]                    // 1 thread load 4 element
    v_mov_b32			v[v_temp2], 			s[s_addr_x+1]
    v_add_co_u32		v[v_addr_x], 			vcc, s[s_addr_x], v[v_temp1]				// v_addr_a = s_addr_a + (bid_x * group_size + tid_x) * 4
    v_addc_co_u32		v[v_addr_x+1], 			vcc, 0, v[v_temp2], vcc

    /* load input */
    global_load_dwordx2	v[v_r0:v_r0+1],			v[v_addr_x:v_addr_x+1], off	offset:0*ELEMT_SIZE
    global_load_dwordx2	v[v_r1:v_r1+1],			v[v_addr_x:v_addr_x+1], off	offset:1*ELEMT_SIZE
    global_load_dwordx2	v[v_r2:v_r2+1],			v[v_addr_x:v_addr_x+1], off	offset:2*ELEMT_SIZE
    global_load_dwordx2	v[v_r3:v_r3+1],			v[v_addr_x:v_addr_x+1], off	offset:3*ELEMT_SIZE
    s_waitcnt           vmcnt(0)

    /* transpose input */
    Transpose			v_r0, v_r1, v_r2, v_r3
    /* PASS 0 */
    FwdRad4B1			v_r0, v_r1, v_r2, v_r3
    Transpose			v_r0, v_r1, v_r2, v_r3

	/* calculate twiddle address */
	/* T W = twiddles[3 + 3*((1*me + 0)%4) + 0/1/2];*/
    v_and_b32			v[v_temp1],				v[v_tid_x], (0x4-1) // v_temp1 = (1*me + 0)%4
    v_mul_u32_u24		v[v_temp1],				v[v_temp1], 0x3     // v_temp1 = 3*((1*me + 0)%4)
    v_add_u32			v[v_temp1],				v[v_temp1], 0x3     // v_temp1 = 3 + 3*((1*me + 0)%4)
    v_lshlrev_b32		v[v_temp1], 			ElemtByteShift, v[v_temp1]
    v_mov_b32			v[v_temp2], 			s[s_addr_w+1]
    v_add_co_u32		v[v_addr_w],			vcc, s[s_addr_w], v[v_temp1]
    v_addc_co_u32		v[v_addr_w+1],			vcc, 0, v[v_temp2], vcc
    /* load twiddle */
    global_load_dwordx2	v[v_w1:v_w1+1],			v[v_addr_w:v_addr_w+1], off	offset:0*ELEMT_SIZE
    global_load_dwordx2	v[v_w2:v_w2+1],			v[v_addr_w:v_addr_w+1], off	offset:1*ELEMT_SIZE
    global_load_dwordx2	v[v_w3:v_w3+1],			v[v_addr_w:v_addr_w+1], off	offset:2*ELEMT_SIZE
    s_waitcnt           vmcnt(0)

    UpdateW				v_r1, v_w1, v_r1
    UpdateW				v_r2, v_w2, v_r2
    UpdateW				v_r3, v_w3, v_r3

    /* PASS 1 */
    FwdRad4B1			v_r0, v_r1, v_r2, v_r3
    Transpose			v_r0, v_r1, v_r2, v_r3

	/* calculate output address */
    v_lshlrev_b32		v[v_temp1],				GroupSizeShift, s[s_bid_x]
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_tid_x], ElemtByteShift
    v_lshlrev_b32		v[v_temp1], 			FftTileShift, v[v_temp1]
    v_mov_b32			v[v_temp2],				s[s_addr_y+1]
    v_add_co_u32		v[v_addr_y],			vcc, s[s_addr_y], v[v_temp1]
    v_addc_co_u32		v[v_addr_y+1],			vcc, 0, v[v_temp2], vcc

	/* store c */
    global_store_dwordx2	v[v_addr_y:v_addr_y+1], v[v_r0:v_r0+1], off offset:0*ELEMT_SIZE
    global_store_dwordx2	v[v_addr_y:v_addr_y+1], v[v_r1:v_r1+1], off offset:1*ELEMT_SIZE
    global_store_dwordx2	v[v_addr_y:v_addr_y+1], v[v_r2:v_r2+1], off offset:2*ELEMT_SIZE
    global_store_dwordx2	v[v_addr_y:v_addr_y+1], v[v_r3:v_r3+1], off offset:3*ELEMT_SIZE

END_PROG:
    s_endpgm


.rodata
.p2align 6
.amdhsa_kernel fft_len16
        .amdhsa_user_sgpr_kernarg_segment_ptr 1	// for kernel argument table
        .amdhsa_system_sgpr_workgroup_id_x 1	// bid_x
        .amdhsa_system_sgpr_workgroup_id_y 1	// bid_y
        .amdhsa_system_sgpr_workgroup_id_z 1	// bid_z
        .amdhsa_system_vgpr_workitem_id 2		// tid_x, tid_y, tid_z

        .amdhsa_next_free_sgpr sgpr_count
        .amdhsa_next_free_vgpr vgpr_count
        .amdhsa_reserve_vcc 1
        .amdhsa_reserve_xnack_mask 0
        .amdhsa_reserve_flat_scratch 0

        .amdhsa_group_segment_fixed_size lds_byte_size
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.altmacro

.macro METADATA group_size, args_size, sgpr_cnt, vgpr_cnt, lds_byte
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: fft_len16
    .symbol: fft_len16.kd
    .sgpr_count: \sgpr_cnt
    .vgpr_count: \vgpr_cnt
    .language: "Assembler"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \args_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_byte
    .private_segment_fixed_size: 0
    .reqd_workgroup_size: [ \group_size, 1, 1 ]
    .max_flat_workgroup_size: \group_size
    .wavefront_size: 64
    .args:
    - { .size: 8, .offset:  0, .value_kind: global_buffer, .value_type: f32, .name: x, .address_space: global, .is_const: true }
    - { .size: 8, .offset:  8, .value_kind: global_buffer, .value_type: f32, .name: w, .address_space: global, .is_const: true }
    - { .size: 8, .offset: 16, .value_kind: global_buffer, .value_type: f32, .name: y, .address_space: global, .is_const: false }
    - { .size: 4, .offset: 24, .value_kind: by_value, .value_type: i32, .name: Len }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %GROUP_SIZE, %args_size, %sgpr_count, %vgpr_count, %lds_byte_size

