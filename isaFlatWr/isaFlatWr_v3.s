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
.globl isaFlatWr
.p2align 6
.type isaFlatWr,@function

/* constanand define */
.set WAVE_SIZE,				(64)
.set WAVE_NUM,				(1)
.set GROUP_NUM,				(2)
.set GROUP_SIZE, 			(WAVE_SIZE * WAVE_NUM)
.set GLOBAL_SIZE,			(GROUP_SIZE * GROUP_NUM)
.set DWORD_SIZE,			(4)

/* kernel argument offset */
.set arg_a_offset,			(0)
.set arg_b_offset,			(8)
.set arg_c_offset,			(16)
.set arg_len_offset,		(24)

.set args_size,				(28)

/* sgpr allocate */
.set s_addr_args,			(0)		// [1:0]
.set s_bid_x,				(2)
.set s_addr_a,				(6)		// [7:6]
.set s_addr_b,				(8)		// [9:8]
.set s_addr_c,				(10)	// [11:10]
.set s_offset,				(12)
.set s_temp1,				(13)
.set s_temp2,				(14)
.set s_buff_dscp,			(16) 	// [19:16]

.set sgpr_count,			(20)

/* vgpr allocate */
.set v_tid_x,				(0)
.set v_addr_a,				(2)		// [3:2]
.set v_addr_b,				(4)		// [5:4]
.set v_addr_c,				(6)		// [7:6]
.set v_a,					(8)
.set v_b,					(9)
.set v_c,					(10)
.set v_offset,				(11)
.set v_temp1,				(12)
.set v_temp2,				(13)

.set vgpr_count,			(14)

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
WaveSizeShift = 0
GroupSizeShift = 0
Dword2ByteShift = 0
Log2Func Dword2ByteShift, DWORD_SIZE
Log2Func WaveSizeShift, WAVE_SIZE
Log2Func GroupSizeShift, GROUP_SIZE

isaFlatWr:
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_a:s_addr_a+1], s[s_addr_args:s_addr_args+1], arg_a_offset
    s_load_dwordx2		s[s_addr_b:s_addr_b+1], s[s_addr_args:s_addr_args+1], arg_b_offset
    s_load_dwordx2		s[s_addr_c:s_addr_c+1], s[s_addr_args:s_addr_args+1], arg_c_offset
    s_waitcnt			lgkmcnt(0)

	/* calculate vector A v_address */
	v_lshlrev_b32		v[v_temp1], GroupSizeShift, s[s_bid_x] 					// temp1 = bid_x * group_size
	v_add_lshl_u32		v[v_offset], v[v_temp1], v[v_tid_x], Dword2ByteShift	// v_offset = (bid_x * group_size + tid_x) * 4
	v_mov_b32			v[v_temp2], s[s_addr_a + 1]
	v_add_co_u32		v[v_addr_a], vcc, s[s_addr_a], v[v_offset]				// v_addr_a = s_addr_a + (bid_x * group_size + tid_x) * 4
	v_addc_co_u32		v[v_addr_a + 1], vcc, 0, v[v_temp2], vcc


	// ------------------------------------------------------
	// global_load_dword	: address = v_addr(64-bit) + immediate(s13) + off
	// flat_load_dword		: address = v_addr(64-bit) + immediate(u12)
	// 32-bit mode for global_load_dword is not supported
	//
	// glc: always miss the L1 and force fetch to L2
	// slc: forced to miss in level 2 texture cache. but if hit L1, it won't access L2 (not work)
	//
	// L1 hit: 128 clk
	// L1 miss, L2 hit: 212 clk
	// L2 miss: about 800 clk
	// ------------------------------------------------------
	global_load_dword	v[v_a], v[v_addr_a:v_addr_a + 1], off
	s_waitcnt			vmcnt(0)
	v_mov_b32			v[v_c], v[v_a]
	
	flat_load_dword		v[v_a], v[v_addr_a:v_addr_a + 1]
	s_waitcnt			vmcnt(0)
	v_add_f32			v[v_c], v[v_c], v[v_a]

	/* calculate vector C address and store c*/
	v_lshlrev_b32		v[v_temp1], GroupSizeShift, s[s_bid_x]
	v_add_lshl_u32		v[v_temp1], v[v_temp1], v[v_tid_x], Dword2ByteShift
	v_mov_b32			v[v_temp2], s[s_addr_c + 1]
	v_add_co_u32		v[v_addr_c], vcc, s[s_addr_c], v[v_temp1]
	v_addc_co_u32		v[v_addr_c + 1], vcc, 0, v[v_temp2], vcc
    flat_store_dword	v[v_addr_c:v_addr_c+1], v[v_c]	

END_PROG:
    s_endpgm


.rodata
.p2align 6
.amdhsa_kernel isaFlatWr
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
  - .name: isaFlatWr
    .symbol: isaFlatWr.kd
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

