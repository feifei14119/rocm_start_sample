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
.globl SmemInstr
.p2align 6
.type SmemInstr,@function

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
WaveSizeShift = 0
GroupSizeShift = 0
Dword2ByteShift = 0
Log2Func Dword2ByteShift, DWORD_SIZE
Log2Func WaveSizeShift, WAVE_SIZE
Log2Func GroupSizeShift, GROUP_SIZE

SmemInstr:    
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_a:s_addr_a+1], s[s_addr_args:s_addr_args+1], arg_a_offset
    s_load_dwordx2		s[s_addr_b:s_addr_b+1], s[s_addr_args:s_addr_args+1], arg_b_offset
    s_load_dwordx2		s[s_addr_c:s_addr_c+1], s[s_addr_args:s_addr_args+1], arg_c_offset
    s_waitcnt			lgkmcnt(0)	

	// ------------------------------------------------------
	// s_load_dword instruction test
	//
	// offset = sgpr or 20bit-immediate
	//
	// s_read_address = A[bid_x]
	// group 0 read A[0]
	// group 1 read A[1]
	// ------------------------------------------------------
	s_lshl_b32			s[s_offset],			s[s_bid_x], Dword2ByteShift
    s_load_dword		s[s_temp1],				s[s_addr_a:s_addr_a+1], s[s_offset]			glc	// addr = addr_a + bid_x * 4
    s_waitcnt			lgkmcnt(0)

	// ------------------------------------------------------
	// s_buffer_load_dword instruction test
	//
	// base_address	: DWORD1[15:0]  + DWORD0[31:0]
	// stride		: DWORD1[29:16]
	// num_records	: DWORD2[31:0]
	// NV			: DWORD2[27]]
	//
	// s_read_addr =  A[4]
	// group 0 read A[4]
	// group 1 read A[4]
	// ------------------------------------------------------
	s_mov_b32			s[s_buff_dscp+0],		s[s_addr_a+0]
	s_mov_b32			s[s_buff_dscp+1],		s[s_addr_a+1]
	s_mov_b32			s[s_buff_dscp+2], 		GLOBAL_SIZE
	s_mov_b32			s[s_buff_dscp+3], 		0x00000000
    s_buffer_load_dword	s[s_temp2],				s[s_buff_dscp:s_buff_dscp+3], 4*4
	s_waitcnt			lgkmcnt(0)
	
	// ------------------------------------------------------
	// s_store_dword instruction test
	//
	// offset = sgpr or 20bit-immediate
	//
	// s_store_address 1 = C[bid_x], s_store_address 2 = C[bid_x + 2]
	// group 0 store C[0] and C[2]
	// group 1 store C[1] and C[3]
	// ------------------------------------------------------
//	s_store_dword		s[s_temp1],				s[s_addr_c:s_addr_c+1], s[s_offset]
//	s_add_u32			s[s_offset],			s[s_offset], 2*4
//	s_store_dword		s[s_temp2],				s[s_addr_c:s_addr_c+1], s[s_offset]
	
	// ------------------------------------------------------
	// use vmem to simulate s_store_dowrd
	// ------------------------------------------------------
	s_mov_b64			exec,					1
	
	v_mov_b32			v[v_addr_c],			s[s_addr_c]
	v_mov_b32			v[v_addr_c+1],			s[s_addr_c+1]
    v_add_co_u32		v[v_addr_c],			vcc, s[s_offset], v[v_addr_c]
    v_addc_co_u32		v[v_addr_c+1],			vcc, 0, v[v_addr_c+1], vcc
	v_mov_b32			v[v_temp1],				s[s_temp1]
	global_store_dword	v[v_addr_c:v_addr_c+1], v[v_temp1], off
	
	v_mov_b32			v[v_addr_c],			s[s_addr_c]
	v_mov_b32			v[v_addr_c+1],			s[s_addr_c+1]
	s_add_u32			s[s_offset],			s[s_offset], 2*4
    v_add_co_u32		v[v_addr_c],			vcc, s[s_offset], v[v_addr_c]
    v_addc_co_u32		v[v_addr_c+1],			vcc, 0, v[v_addr_c+1], vcc
	v_mov_b32			v[v_temp2],				s[s_temp2]
	global_store_dword	v[v_addr_c:v_addr_c+1], v[v_temp2], off
	
END_PROG:
    s_endpgm


.rodata
.p2align 6
.amdhsa_kernel SmemInstr
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
  - .name: SmemInstr
    .symbol: SmemInstr.kd
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

