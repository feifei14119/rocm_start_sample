import subprocess
import sys, os, re
import shutil

Target = "vectAdd2.out"

def execCmd(cmd):		
	r = os.popen(cmd)  
	text = r.read() 
	print(text)
	r.close()  
	return text 
	
def BuildTarget():
	global KernelType
	
	if os.path.exists("./" + Target):
		os.remove("./" + Target)
		
	cmd = 'hipcc ../VectorAddMain.cpp ' + '-D__HIP_PLATFORM_HCC__=  -I/opt/rocm/hip/include -I/opt/rocm/llvm/bin/../lib/clang/11.0.0 -I/opt/rocm/hsa/include -D__HIP_ROCclr__ -O0 -w -std=c++11 -o ' + Target
	print(cmd)
	text = execCmd(cmd)
	print(text)
	
	return
	
def RunTarget():
	cmd = "./" + Target
	print(cmd)
	execCmd(cmd)


def RunBuild():
	global KernelType
	global ObjectVersion
	global Compiler
	
	if(os.path.exists("./out")):
		shutil.rmtree("./out")
	os.mkdir("./out")
	os.chdir("./out")
		
	BuildTarget()
	RunTarget()

if __name__ == '__main__':
	RunBuild()
	exit()
	
	
