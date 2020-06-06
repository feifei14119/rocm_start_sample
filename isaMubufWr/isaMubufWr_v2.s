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
.globl isaMubufWr
.p2align 8
.type isaMubufWr,@function
.amdgpu_hsa_kernel isaMubufWr

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
.set s_bid_y,				(3)
.set s_bid_z,				(4)

.set s_addr_a,				(6)		// [7:6]
.set s_addr_b,				(8)		// [9:8]
.set s_addr_c,				(10)	// [11:10]

.set s_temp1,				(12)
.set s_temp2,				(13)

.set s_buff_dscp,			(16)	// [19:16]
.set s_offset,				(20)

.set sgpr_count,			(22)

/* vgpr allocate */
.set v_tid_x,				(0)
.set v_tid_y,				(1)
.set v_tid_z,				(2)

.set v_addr_a,				(4)		// [5:4]
.set v_addr_b,				(6)		// [7:6]
.set v_addr_c,				(8)		// [9:8]
.set v_a,					(10)
.set v_b,					(11)
.set v_c,					(12)

.set v_temp1,				(13)
.set v_temp2,				(14)

.set v_idx,					(15)
.set v_offset,				(16)
.set v_offset_idx,			(15)	// [v_offset:v_idx] = [16:15]

.set vgpr_count,			(17)

/* lds allocate */
.set lds_byte_size,			(DWORD_SIZE * WAVE_SIZE * WAVE_NUM)

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

isaMubufWr:
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
	// buffer_load_dword instruction test
	//
	// buffer_load_dword 	v_dst, 	v_offset_idx_2, s_dscp_4, s_offset 	idx_en offset_en imm_offset glc slc lds
	//
	// -------------- s_dscp_4 ------------------
	// base_addr	: DWORD1[15:0] + DWORD0[31:0]
	// stride		: DWORD1[29:16]		; byte
	// num_records	: DWORD2[31:0]		; stride
	// data_format	: DWORD3[18:15] 	; 1=8bit, 2=16bit, 4=32bit
	// add_tid_en	: DWORD3[23]
	// ------------- v_offset_idx_2 -------------
	// v_offset_idx_2 = [v_offset,v_idx]
	// --------------- address ------------------
	// global_address = base_addr(dscp) + s_offset + stride*index + offset
	//			index = v_idx(?idx_en) + tid_in_wave(?add_tid_en(dscp))
	//		   offset = v_offset(?offset_en) + imm_offset
	// lds_address = M0 + imm_offset + 4 * tid_in_wave
	// ------------------------------------------
	//
	// stride = 4 Byte
	// num_records = GLOBAL_SIZE
	// data_format = float
	// tid_enable = 1
	// global_address = base_addr +   s_offset 	 + stride*(tid_in_wave +  v_idx ) + (v_offset + imm_offset)
	//  			  = s_addr_a  + 4*group_size +   4   *(     ?      + v_tid_x) + (   4*4   +     0     )		
	// ------------------------------------------------------
	s_mov_b32			s[s_buff_dscp+0],		s[s_addr_a+0]
	s_mov_b32			s[s_buff_dscp+1],		s[s_addr_a+1]
	s_or_b32			s[s_buff_dscp+1],		s[s_buff_dscp+1], 0x00040000 				// stride = 4 = 4Byte
	s_mov_b32			s[s_buff_dscp+2],		GLOBAL_SIZE									// num_rcd = GLOBAL_SIZE
	s_mov_b32			s[s_buff_dscp+3],		0x00020000									// data_format = 4 = 32-bit
	s_or_b32			s[s_buff_dscp+3],		s[s_buff_dscp+3], 0x00800000 				// add_tid_en = 1; why don't work?
	
    s_lshl_b32			s[s_temp1],				s[s_bid_x], GroupSizeShift
    s_lshl_b32			s[s_temp1],				s[s_temp1], Dword2ByteShift
	s_mov_b32			s[s_offset],			s[s_temp1]
	v_mov_b32			v[v_idx],				v[v_tid_x]
	v_mov_b32			v[v_offset],			4*4											// to test over GLOBAL_SIZE
	buffer_load_dword	v[v_a],					v[v_idx:v_offset], s[s_buff_dscp:s_buff_dscp+3], s[s_offset] 	idxen offen offset:0x0
	s_waitcnt			vmcnt(0)	
	
	/* calculate vector C address */
    v_lshlrev_b32		v[v_temp1],				GroupSizeShift, s[s_bid_x]
    v_add_lshl_u32		v[v_temp1],				v[v_temp1], v[v_tid_x], Dword2ByteShift
    v_mov_b32			v[v_temp2],				s[s_addr_c+1]
    v_add_co_u32		v[v_addr_c],			vcc, s[s_addr_c], v[v_temp1]
    v_addc_co_u32		v[v_addr_c+1],			vcc, 0, v[v_temp2], vcc	
	
	/* store c */
	v_mov_b32			v[v_c],					v[v_a]
    global_store_dword	v[v_addr_c:v_addr_c+1], v[v_c], off	
	
END_PROG:
    s_endpgm

.macro METADATA group_size, args_size
    .amd_amdgpu_hsa_metadata
    { Version: [1, 0],
      Kernels :
        - { Name: isaMubufWr, SymbolName: isaMubufWr, Language: Assembler, LanguageVersion: [ 1, 2 ],
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
