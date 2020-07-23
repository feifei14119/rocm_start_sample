Rocm Start Sample
====
rocm-hip on amdgpu入门示例

---
## 1. 使用：
run  
**"python build.py [*hip/asm*] [*v2/v3*] [*llvm/hcc*]"**  
under each project folder, the exacutable will generated under ./out path.  

----
## 2. 示例：
### 2.1 runtime 示例：
>>[1. 设备查询](https://github.com/feifei14119/rocm_start_sample/tree/master/DeviceInfo)  
>>[2. 带宽测试](https://github.com/feifei14119/rocm_start_sample/tree/master/MemBandwidth)  
>>[3. 向量加模板](https://github.com/feifei14119/rocm_start_sample/tree/master/VectorAdd) 
  
### 2.2 hip kernel function 示例：
>>[1. 计时函数](https://github.com/feifei14119/rocm_start_sample/tree/master/hipClock)  
>>[2. 内联汇编](https://github.com/feifei14119/rocm_start_sample/tree/master/hipInlineAsm)  
>>[3. 原子操作](https://github.com/feifei14119/rocm_start_sample/tree/master/hipAtomic)  
>>[4. shuffle操作](https://github.com/feifei14119/rocm_start_sample/tree/master/hipShuffle)  
>>[5. vote操作](https://github.com/feifei14119/rocm_start_sample/tree/master/hipVote)  
  
### 2.3 ISA 示例：
>>[1. smem读写](https://github.com/feifei14119/rocm_start_sample/tree/master/isaSmemWr)  
>>[2. flat读写](https://github.com/feifei14119/rocm_start_sample/tree/master/isaFlatWr)  
>>[3. mubuf读写](https://github.com/feifei14119/rocm_start_sample/tree/master/isaMubufWr)  
>>[4. lds读写](https://github.com/feifei14119/rocm_start_sample/tree/master/isaLdsWr)  
>>[5. group间条件跳转](https://github.com/feifei14119/rocm_start_sample/tree/master/isaSbranch)  
>>[6. thread间条件执行](https://github.com/feifei14119/rocm_start_sample/tree/master/isaVbranch)  
>>[7. packed float16指令](https://github.com/feifei14119/rocm_start_sample/tree/master/isaPackedFp16)  
  
### 2.4 性能优化示例:  
>>[1. 指令发射](https://github.com/feifei14119/rocm_start_sample/tree/master/instrIssue)  
