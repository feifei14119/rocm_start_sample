import subprocess
import sys, os, re
import shutil

KernelType = "ASM_KERNEL"
Target = "vector_add.out"
KernelBin = "VectorAdd.bin"

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
		
	cmd = 'hipcc ../main.cpp -D"' + KernelType + '" -O0 -w -std=c++11 -o ' + Target
	print(cmd)
	text = execCmd(cmd)
	print(text)
	
	return
	
def RunTarget():
	if os.path.exists("../" + KernelBin):
		os.remove("../" + KernelBin)
	
	cmd = "./" + Target
	print(cmd)
	execCmd(cmd)
	
if __name__ == '__main__':
	global KernelType
	
	if(os.path.exists("./out")):
		shutil.rmtree("./out")
	os.mkdir("./out")
	os.chdir("./out")

	if(len(sys.argv) > 1):
		if(sys.argv[1] == "asm"):
			KernelType = "ASM_KERNEL"
		if(sys.argv[1] == 'hip'):
			KernelType = "HIP_KERNEL"
			
	BuildTarget()
	RunTarget()
	exit()
	
	
