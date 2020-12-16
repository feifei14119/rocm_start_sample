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
.globl isaDpp
.p2align 6
.type isaDpp,@function

/* constanand define */
.set WAVE_SIZE,				(64)
.set SIMD_NUM,				(1)
.set GROUP_NUM,				(1)
.set GROUP_SIZE, 			(WAVE_SIZE * SIMD_NUM)
.set GLOBAL_SIZE,			(GROUP_SIZE * GROUP_NUM)
.set DWORD_SIZE,			(4)

/* kernel argument offset */
.set arg_a_offset,			(0)
.set arg_b_offset,			(8)
.set arg_c_offset,			(16)
.set arg_len_offset,		(24)

.set args_size,				(28)

/* sgpr allocate */
.set s_addr_args,			(0)
.set s_bid_x,				(2)
.set s_addr_a,				(6)
.set s_addr_b,				(8)
.set s_addr_c,				(10)

.set sgpr_count,			(11 + 1)

/* vgpr allocate */
.set v_tid_x,				(0)
.set v_addr_a,				(2)
.set v_addr_b,				(4)
.set v_addr_c,				(6)
.set v_a,					(8)
.set v_b,					(9)
.set v_c,					(10)
.set v_temp1,				(11)
.set v_temp2,				(12)

.set vgpr_count,			(13)

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

/* variable  declare */
GroupSizeShift = 0
Dword2ByteShift = 0
Log2Func Dword2ByteShift, DWORD_SIZE
Log2Func GroupSizeShift, GROUP_SIZE

isaDpp:    
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_a:s_addr_a+1], s[s_addr_args:s_addr_args+1], arg_a_offset
    s_load_dwordx2		s[s_addr_b:s_addr_b+1], s[s_addr_args:s_addr_args+1], arg_b_offset
    s_load_dwordx2		s[s_addr_c:s_addr_c+1], s[s_addr_args:s_addr_args+1], arg_c_offset
    s_waitcnt			lgkmcnt(0)

	/* calculate vector A address */
    v_lshlrev_b32		v[v_temp1], 			GroupSizeShift, s[s_bid_x] 					// temp1 = bid_x * group_size
    v_add_lshl_u32		v[v_temp1], 			v[v_temp1], v[v_tid_x], Dword2ByteShift		// temp1 = (bid_x * group_size + tid_x) * 4
    v_mov_b32			v[v_temp2], 			s[s_addr_a+1]
    v_add_co_u32		v[v_addr_a], 			vcc, s[s_addr_a], v[v_temp1]				// v_addr_a = s_addr_a + (bid_x * group_size + tid_x) * 4
    v_addc_co_u32		v[v_addr_a+1], 			vcc, 0, v[v_temp2], vcc

	/* calculate vector B address */
    v_lshlrev_b32		v[v_temp1],				GroupSizeShift, s[s_bid_x]
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_tid_x], Dword2ByteShift
    v_mov_b32			v[v_temp2],				s[s_addr_b+1]
    v_add_co_u32		v[v_addr_b],			vcc, s[s_addr_b], v[v_temp1]
    v_addc_co_u32		v[v_addr_b+1],			vcc, 0, v[v_temp2], vcc

	/* calculate vector C address */
    v_lshlrev_b32		v[v_temp1],				GroupSizeShift, s[s_bid_x]
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_tid_x], Dword2ByteShift
    v_mov_b32			v[v_temp2],				s[s_addr_c+1]
    v_add_co_u32		v[v_addr_c],			vcc, s[s_addr_c], v[v_temp1]
    v_addc_co_u32		v[v_addr_c+1],			vcc, 0, v[v_temp2], vcc

	/* load a/b */
    global_load_dword	v[v_a],					v[v_addr_a:v_addr_a+1], off					// a = * v_addr_a
    global_load_dword	v[v_b],					v[v_addr_b:v_addr_b+1], off
	s_waitcnt			vmcnt(0)

    /* 
     * c += a
     * one row = 16 thread
     * one bank = 4 thread
     *
     *       |     bank0      |      bank1     |      bank2     |      bank3     |
     * row0: | 00, 01, 02, 03,| 04, 05, 06, 07,| 08, 09, 10, 11,| 12, 13, 14, 15 |
     * row1: | 16, 17, 18, 19,| 20, 21, 22, 23,| 24, 25, 26, 27,| 28, 29, 30, 31 |
     * row2: | 32, 33, 34, 35,| 36, 37, 38, 39,| 40, 41, 42, 43,| 44, 45, 46, 47 |
     * row3: | 48, 49, 50, 51,| 52, 53, 54, 55,| 56, 57, 58, 59,| 60, 61, 62, 63 |
     *
     * v_mov_b32_dpp		v[v_c],					v[v_a]          quad_perm:[3,2,1,0]
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_mirror
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_share:1 // not work
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_xmask:1 // not work
     * v_mov_b32_dpp		v[v_c],					v[v_a]          wave_shl:1
     * v_mov_b32_dpp		v[v_c],					v[v_a]          wave_shr:1
     * v_mov_b32_dpp		v[v_c],					v[v_a]          wave_rol:1
     * v_mov_b32_dpp		v[v_c],					v[v_a]          wave_ror:1
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_shl:1~15
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_shr:1~15
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_ror:1~15
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_bcast:15
     * v_mov_b32_dpp		v[v_c],					v[v_a]          row_bcast:31
     */
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_shr:1 bound_ctrl:0
    s_nop				1
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_shr:2 bound_ctrl:0
    s_nop				1
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_shr:4 bound_ctrl:0
    s_nop				1
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_shr:8 bound_ctrl:0
    s_nop				1
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_bcast:15 row_mask:0b1010
    s_nop				1
    v_add_f32_dpp		v[v_a],					v[v_a], v[v_a]	row_bcast:31 row_mask:0b1100
    s_nop				1
    v_mov_b32_dpp		v[v_c],					v[v_a]          wave_ror:1

	/* store c */
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c], off									// * v_addr_c = c
END_PROG:
    s_endpgm


.rodata
.p2align 6
.amdhsa_kernel isaDpp
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
  - .name: isaDpp
    .symbol: isaDpp.kd
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
    - { .size: 8, .offset:  0, .value_kind: global_buffer, .value_type: f32, .name: A, .address_space: global, .is_const: true }
    - { .size: 8, .offset:  8, .value_kind: global_buffer, .value_type: f32, .name: B, .address_space: global, .is_const: true }
    - { .size: 8, .offset: 16, .value_kind: global_buffer, .value_type: f32, .name: C, .address_space: global, .is_const: false }
    - { .size: 4, .offset: 24, .value_kind: by_value, .value_type: i32, .name: Len }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %GROUP_SIZE, %args_size, %sgpr_count, %vgpr_count, %lds_byte_size

