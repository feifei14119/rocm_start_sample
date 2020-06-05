/*
 * 0. modify hsa_code_object_isa
 * 1. modify kernel name
 * 2. modify GROUP_NUM or GROUP_SIZE
 * 3. modify kernel arguments offset in constanand define and amd_amdgpu_hsa_metadata, and args_size value
 * 4. modify sgpr allocate and sgpr_count value
 * 5. modify vgpr allocate and vgpr_count value
 * 6. modify lds allocate and lds_byte_size value
*/
.hsa_code_object_version 2, 1
.hsa_code_object_isa 9, 0, 6, "AMD", "AMDGPU"

.text
.globl isaSmemWr
.p2align 8
.type isaSmemWr,@function
.amdgpu_hsa_kernel isaSmemWr

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

isaSmemWr:
    .amd_kernel_code_t
        amd_code_version_major = 1
        amd_code_version_minor = 1
        amd_machine_kind = 1
        amd_machine_version_major = 8
        amd_machine_version_minor = 0
        amd_machine_version_stepping = 3
        kernarg_segment_alignment = 4
        group_segment_alignment = 4
        private_segment_alignment = 4
        wavefront_size = 6					// 2^6 = 64
        call_convention = -1
        is_ptr64 = 1
        float_mode = 192
		
        enable_vgpr_workitem_id = 2			// tid_x, tid_y, tid_z
        enable_sgpr_workgroup_id_x = 1		// bid_x
        enable_sgpr_workgroup_id_y = 1		// bid_y
        enable_sgpr_workgroup_id_z = 1		// bid_z
		
        enable_sgpr_kernarg_segment_ptr = 1
        user_sgpr_count = 2 				// for kernel argument table
        kernarg_segment_byte_size = args_size
		
        wavefront_sgpr_count = sgpr_count
        workitem_vgpr_count = vgpr_count
        granulated_wavefront_sgpr_count = (sgpr_count - 1)/8
        granulated_workitem_vgpr_count = (vgpr_count - 1)/4
		
        workgroup_group_segment_byte_size = lds_byte_size
    .end_amd_kernel_code_t
	
START_PROG:
	/* load kernel arguments */
    s_load_dwordx2		s[s_addr_a:s_addr_a+1], s[s_addr_args:s_addr_args+1], arg_a_offset
    s_load_dwordx2		s[s_addr_b:s_addr_b+1], s[s_addr_args:s_addr_args+1], arg_b_offset
    s_load_dwordx2		s[s_addr_c:s_addr_c+1], s[s_addr_args:s_addr_args+1], arg_c_offset
    s_waitcnt			lgkmcnt(0)	

	// ------------------------------------------------------
	// s_load_dword instruction test
	//
	// offset = sgpr or u20-immediate
	// glc: miss L1 cache
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

.macro METADATA group_size, args_size
    .amd_amdgpu_hsa_metadata
    { Version: [1, 0],
      Kernels :
        - { Name: isaSmemWr, SymbolName: isaSmemWr, Language: Assembler, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \group_size, 1, 1 ] }
            CodeProps:
              { KernargSegmentSize: \args_size, GroupSegmentFixedSize : 0, PrivateSegmentFixedSize : 0, KernargSegmentAlign : 8, WavefrontSize : 64, MaxFlatWorkGroupSize : \group_size }
            Args:
            - { Name: d_a, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
            - { Name: d_b, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
            - { Name: d_c, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
            - { Name: len, Size: 4, Align: 4, ValueKind: ByValue, ValueType: U32, TypeName: 'int', AddrSpaceQual: Global, IsConst: true }
          }
    }
    .end_amd_amdgpu_hsa_metadata
.endm

.altmacro
.macro METADATA_WRAPPER group_size, args_size
    METADATA %\group_size, %\args_size
.endm

METADATA_WRAPPER GROUP_SIZE, args_size
