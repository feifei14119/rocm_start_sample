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
.globl isaMfma
.p2align 6
.type isaMfma,@function

/* constanand define */
.set WAVE_SIZE,				(64)
.set SIMD_NUM,				(1)
.set GROUP_NUM,				(1)
.set GROUP_SIZE, 			(WAVE_SIZE * SIMD_NUM)
.set GLOBAL_SIZE,			(GROUP_SIZE * GROUP_NUM)
.set ELMT_SIZE,			    (4) // float
.set M,				        (32)
.set N,				        (32)
.set K,				        (128)

/* kernel argument offset */
.set arg_a_offset,			(0)
.set arg_b_offset,			(8)
.set arg_c_offset,			(16)
.set arg_m_offset,		    (24)
.set arg_n_offset,		    (28)
.set arg_k_offset,		    (32)

.set args_size,				(36)

/* sgpr allocate */
.set s_addr_args,			(0)
.set s_bid_x,				(2)
.set s_addr_a,				(6)
.set s_addr_b,				(8)
.set s_addr_c,				(10)
.set s_m,   				(12)
.set s_n,   				(13)
.set s_k,   				(14)

.set sgpr_count,			(100 + 1)

/* vgpr allocate */
.set v_tid_x,				(0)
.set v_addr_a,				(2)
.set v_addr_b,				(4)
.set v_addr_c,				(6)
.set v_step_a,				(8)
.set v_step_b,				(9)
.set v_temp1,				(12)
.set v_temp2,				(13)
.set v_a,					(14)
.set v_b,					(15)
.set v_c,					(16)

.set vgpr_count,			(31)

/* accgpr allocate */
.set acc_d,				    (0)

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
ElemntByteShift = 0
MFMA_M = 32
MFMA_N = 32
MFMA_K = 2
Log2Func ElemntByteShift, ELMT_SIZE
Log2Func GroupSizeShift, GROUP_SIZE
Log2Func MFMA_MShift, MFMA_M
Log2Func MFMA_NShift, MFMA_N
Log2Func MFMA_KShift, MFMA_K

isaMfma:    
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_a:s_addr_a+1], s[s_addr_args:s_addr_args+1], arg_a_offset
    s_load_dwordx2		s[s_addr_b:s_addr_b+1], s[s_addr_args:s_addr_args+1], arg_b_offset
    s_load_dwordx2		s[s_addr_c:s_addr_c+1], s[s_addr_args:s_addr_args+1], arg_c_offset
    s_load_dword		s[s_m], s[s_addr_args:s_addr_args+1], arg_m_offset
    s_load_dword		s[s_n], s[s_addr_args:s_addr_args+1], arg_n_offset
    s_load_dword		s[s_k], s[s_addr_args:s_addr_args+1], arg_k_offset
    s_waitcnt			lgkmcnt(0)

	/*
     * calculate A address 
     * k = tid_x / MFMA_M
     * m = tid_x % MFMA_M
     * addr_a = M * k + m
     * step_a = M * MFMA_K
     */
    v_and_b32		    v[v_temp1], 			v[v_tid_x], MFMA_M-1 			// temp1 = tid_x % 32
    v_lshrrev_b32		v[v_temp2],				MFMA_MShift, v[v_tid_x]         // temp2 = tid_x / 32
    v_mul_u32_u24_e64   v[v_temp2],             v[v_temp2], M                   // temp2 = (tid_x / 32) * M
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_temp2], ElemntByteShift // temp1 = (BYTE)(k * M + m)
    v_mov_b32			v[v_temp2], 			s[s_addr_a+1]
    v_add_co_u32		v[v_addr_a], 			vcc, s[s_addr_a], v[v_temp1]
    v_addc_co_u32		v[v_addr_a+1], 			vcc, 0, v[v_temp2], vcc
    v_mov_b32			v[v_step_a], 			0x0 + ELMT_SIZE * M*MFMA_K

	/*
     * calculate B address 
     * k = tid_x / MFMA_N
     * m = tid_x % MFMA_N
     * addr_b = N * k + n
     * step_b = N * MFMA_K
     */
    v_and_b32		    v[v_temp1], 			v[v_tid_x], MFMA_N-1
    v_lshrrev_b32		v[v_temp2],				MFMA_NShift, v[v_tid_x]
    v_mul_u32_u24_e64   v[v_temp2],             v[v_temp2], N
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_temp2], ElemntByteShift
    v_mov_b32			v[v_temp2], 			s[s_addr_b+1]
    v_add_co_u32		v[v_addr_b], 			vcc, s[s_addr_b], v[v_temp1]
    v_addc_co_u32		v[v_addr_b+1], 			vcc, 0, v[v_temp2], vcc
    v_mov_b32			v[v_step_b], 			0x0 + ELMT_SIZE * N*MFMA_K

    /* init accvgpr d */
    v_accvgpr_write     acc[acc_d+0],            0
    v_accvgpr_write     acc[acc_d+1],            0
    v_accvgpr_write     acc[acc_d+2],            0
    v_accvgpr_write     acc[acc_d+3],            0
    v_accvgpr_write     acc[acc_d+4],            0
    v_accvgpr_write     acc[acc_d+5],            0
    v_accvgpr_write     acc[acc_d+6],            0
    v_accvgpr_write     acc[acc_d+7],            0
    v_accvgpr_write     acc[acc_d+8],            0
    v_accvgpr_write     acc[acc_d+9],            0
    v_accvgpr_write     acc[acc_d+10],            0
    v_accvgpr_write     acc[acc_d+11],            0
    v_accvgpr_write     acc[acc_d+12],            0
    v_accvgpr_write     acc[acc_d+13],            0
    v_accvgpr_write     acc[acc_d+14],            0
    v_accvgpr_write     acc[acc_d+15],            0

	/* load a/b */
	.rept (K/MFMA_K)
		global_load_dword	v[v_a],					v[v_addr_a:v_addr_a+1], off
		global_load_dword	v[v_b],					v[v_addr_b:v_addr_b+1], off
		s_waitcnt			vmcnt(0)
				
		/* do matrix mul */
		v_mfma_f32_32x32x2f32 acc[acc_d:acc_d+15],	v[v_a],v[v_b],acc[acc_d:acc_d+15]
		s_nop				64
		
		v_add_co_u32		v[v_addr_a], 			vcc, v[v_addr_a], v[v_step_a]
		v_addc_co_u32		v[v_addr_a+1], 			vcc, 0, v[v_addr_a+1], vcc
		v_add_co_u32		v[v_addr_b], 			vcc, v[v_addr_b], v[v_step_b]
		v_addc_co_u32		v[v_addr_b+1], 			vcc, 0, v[v_addr_b+1], vcc
	.endr
			
	
	/* read result */	
    v_accvgpr_read      v[v_c+0],                acc[acc_d+0]
    s_nop               4
    v_accvgpr_read      v[v_c+1],                acc[acc_d+1]
    s_nop               4
    v_accvgpr_read      v[v_c+2],                acc[acc_d+2]
    s_nop               4
    v_accvgpr_read      v[v_c+3],                acc[acc_d+3]
    s_nop               4
    v_accvgpr_read      v[v_c+4],                acc[acc_d+4]
    s_nop               4
    v_accvgpr_read      v[v_c+5],                acc[acc_d+5]
    s_nop               4
    v_accvgpr_read      v[v_c+6],                acc[acc_d+6]
    s_nop               4
    v_accvgpr_read      v[v_c+7],                acc[acc_d+7]
    s_nop               4
    v_accvgpr_read      v[v_c+8],                acc[acc_d+8]
    s_nop               4
    v_accvgpr_read      v[v_c+9],                acc[acc_d+9]
    s_nop               4
    v_accvgpr_read      v[v_c+10],                acc[acc_d+10]
    s_nop               4
    v_accvgpr_read      v[v_c+11],                acc[acc_d+11]
    s_nop               4
    v_accvgpr_read      v[v_c+12],                acc[acc_d+12]
    s_nop               4
    v_accvgpr_read      v[v_c+13],                acc[acc_d+13]
    s_nop               4
    v_accvgpr_read      v[v_c+14],                acc[acc_d+14]
    s_nop               4
    v_accvgpr_read      v[v_c+15],                acc[acc_d+15]
    s_nop               4
	
	/* 
     * calculate c address 
	 * n = tid_x % MFMA_N
	 * lane_id = tid_x / MFMA_N
	 * m = lane_id * 4
	 * addr_c = M * n + m
	 */
    v_and_b32		    v[v_temp1], 			v[v_tid_x], MFMA_N-1				// temp1 = n = tid_x % 32	
    v_lshrrev_b32		v[v_temp2],				MFMA_NShift, v[v_tid_x]             // temp2 = lane_id = tid_x / 32	
    v_lshlrev_b32		v[v_temp2],				2, v[v_temp2]             			// temp2 = m = lane_id * 4	
    v_mul_u32_u24_e64   v[v_temp1],             v[v_temp1], M						// temp2 = M * n
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_temp2], ElemntByteShift // temp1 = (BYTE)(M * n + m)
    v_mov_b32			v[v_temp2], 			s[s_addr_c+1]
    v_add_co_u32		v[v_addr_c], 			vcc, s[s_addr_c], v[v_temp1]
    v_addc_co_u32		v[v_addr_c+1], 			vcc, 0, v[v_temp2], vcc
	
	/* store result */
	flat_store_dwordx4	v[v_addr_c:v_addr_c+1], v[v_c+0:v_c+3] offset:ELMT_SIZE*8*0
	flat_store_dwordx4	v[v_addr_c:v_addr_c+1], v[v_c+4:v_c+7] offset:ELMT_SIZE*8*1
	flat_store_dwordx4	v[v_addr_c:v_addr_c+1], v[v_c+8:v_c+11] offset:ELMT_SIZE*8*2
	flat_store_dwordx4	v[v_addr_c:v_addr_c+1], v[v_c+12:v_c+15] offset:ELMT_SIZE*8*3

.if (0)
	// ---------------------- debug ---------------------------------------
	/* 
     * calculate debug address 
     * every thread store 16 elements
     * line_id = tid_x
     * addr_dbg = 16 * line_id
     */
    v_mul_u32_u24       v[v_temp1],             v[v_tid_x], 0x10             // temp1 = tid_x * 16
    v_lshlrev_b32		v[v_temp1],				ElemntByteShift, v[v_temp1]  // temp1 = (BYTE)(k * M + m)
    v_mov_b32			v[v_temp2],				s[s_addr_c+1]
    v_add_co_u32		v[v_addr_c],			vcc, s[s_addr_c], v[v_temp1]
    v_addc_co_u32		v[v_addr_c+1],			vcc, 0, v[v_temp2], vcc
		
	/* store debug */	
	v_cvt_f32_u32		v[v_c], v[v_tid_x]
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+0], off offset:ELMT_SIZE*0
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+1], off offset:ELMT_SIZE*1
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+2], off offset:ELMT_SIZE*2
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+3], off offset:ELMT_SIZE*3
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+4], off offset:ELMT_SIZE*4
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+5], off offset:ELMT_SIZE*5
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+6], off offset:ELMT_SIZE*6
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+7], off offset:ELMT_SIZE*7
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+8], off offset:ELMT_SIZE*8
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+9], off offset:ELMT_SIZE*9
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+10], off offset:ELMT_SIZE*10
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+11], off offset:ELMT_SIZE*11
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+12], off offset:ELMT_SIZE*12
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+13], off offset:ELMT_SIZE*13
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+14], off offset:ELMT_SIZE*14
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c+15], off offset:ELMT_SIZE*15
.endif
END_PROG:
    s_endpgm


.rodata
.p2align 6
.amdhsa_kernel isaMfma
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
  - .name: isaMfma
    .symbol: isaMfma.kd
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
    - { .size: 4, .offset: 24, .value_kind: by_value, .value_type: i32, .name: m }
    - { .size: 4, .offset: 28, .value_kind: by_value, .value_type: i32, .name: n }
    - { .size: 4, .offset: 32, .value_kind: by_value, .value_type: i32, .name: k }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %GROUP_SIZE, %args_size, %sgpr_count, %vgpr_count, %lds_byte_size

