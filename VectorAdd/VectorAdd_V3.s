.text
.globl VectorAdd
.p2align 8
.type VectorAdd,@function

VectorAdd:    
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

.rodata
.p2align 6
.amdhsa_kernel VectorAdd
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 2
    .amdhsa_next_free_vgpr 19
    .amdhsa_next_free_sgpr 14
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: VectorAdd
    .symbol: VectorAdd.kd
    .sgpr_count: 14
    .vgpr_count: 19
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: d_a, .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: d_b, .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: d_c, .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: len, .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
	
.end_amdgpu_metadata
