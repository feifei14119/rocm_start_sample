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
.globl isaFlatWr
.p2align 8
.type isaFlatWr,@function
.amdgpu_hsa_kernel isaFlatWr

/* constanand define */
.set WAVE_SIZE,				(64)
.set WAVE_NUM,				(1)
.set GROUP_NUM,				(60)
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

.macro METADATA group_size, args_size
    .amd_amdgpu_hsa_metadata
    { Version: [1, 0],
      Kernels :
        - { Name: isaFlatWr, SymbolName: isaFlatWr, Language: Assembler, LanguageVersion: [ 1, 2 ],
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
