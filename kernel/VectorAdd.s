.hsa_code_object_version 2, 1
.hsa_code_object_isa 9, 0, 6, "AMD", "AMDGPU"

.text
.globl VectorAdd
.p2align 8
.type VectorAdd,@function
.amdgpu_hsa_kernel VectorAdd

VectorAdd:
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
        wavefront_size = 6
        call_convention = -1
        enable_sgpr_kernarg_segment_ptr = 1
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_sgpr_workgroup_id_z = 1
        enable_vgpr_workitem_id = 2
        is_ptr64 = 1
        float_mode = 192
        granulated_wavefront_sgpr_count = 2
        granulated_workitem_vgpr_count = 3
        user_sgpr_count = 2
        wavefront_sgpr_count = 18
        workitem_vgpr_count = 13
        kernarg_segment_byte_size = 44
        workgroup_group_segment_byte_size = 0
    .end_amd_kernel_code_t
    
START_PROG:
    s_load_dwordx2                              s[6:7], s[0:1], 0
    s_load_dwordx2                              s[8:9], s[0:1], 8
    s_load_dwordx2                              s[10:11], s[0:1], 16
    s_waitcnt                                   lgkmcnt(0)
	
    v_lshlrev_b32                               v[11], 8, s[2]                           
    v_add_lshl_u32                              v[11], v[11], v[0], 2                    
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[12], s[7]
    v_add_co_u32                                v[2], vcc, s[6], v[11]                   
    v_addc_co_u32                               v[3], vcc, 0, v[12], vcc
    v_lshlrev_b32                               v[11], 8, s[2]                           
    v_add_lshl_u32                              v[11], v[11], v[0], 2                    
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[12], s[9]
    v_add_co_u32                                v[4], vcc, s[8], v[11]                   
    v_addc_co_u32                               v[5], vcc, 0, v[12], vcc
    v_lshlrev_b32                               v[11], 8, s[2]                           
    v_add_lshl_u32                              v[11], v[11], v[0], 2                    
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[12], s[11]
    v_add_co_u32                                v[6], vcc, s[10], v[11]                  
    v_addc_co_u32                               v[7], vcc, 0, v[12], vcc
    global_load_dword                           v[8], v[2:3], off                         
    global_load_dword                           v[9], v[4:5], off                         
    s_waitcnt                                   vmcnt(0)
    v_add_f32                                   v[10], v[8], v[9]                        
    global_store_dword                          v[6:7], v[10], off 
END_PROG:
    s_endpgm

.amd_amdgpu_hsa_metadata
{ Version: [1, 0],
  Kernels :
    - { Name: VectorAdd,
        SymbolName: VectorAdd,
        Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs: { ReqdWorkGroupSize: [ 256, 1, 1 ] }
        CodeProps: { KernargSegmentSize: 28, GroupSegmentFixedSize : 0, PrivateSegmentFixedSize : 0, KernargSegmentAlign : 8, WavefrontSize : 64, MaxFlatWorkGroupSize : 64 }
        Args:
        - { Name: d_a, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_b, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_c, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
        - { Name: len, Size: 4, Align: 4, ValueKind: ByValue, ValueType: U32, TypeName: 'int', AddrSpaceQual: Global, IsConst: true }
      }
}
.end_amd_amdgpu_hsa_metadata

