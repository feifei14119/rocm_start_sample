import subprocess
import sys, os, re
import shutil

KernelType = "HIP_KERNEL"
Target = "atomic_func.out"

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
		
	cmd = 'hipcc ../main.cpp ../../common/utils.cpp -D"' + KernelType + '" -O0 -w -std=c++11 -o ' + Target
	print(cmd)
	text = execCmd(cmd)
	print(text)
	
	return
	
def RunTarget():
	# remove kernel bin
	if os.path.exists("../*.bin"):
		os.remove("../*.bin")
	
	cmd = "./" + Target
	print(cmd)
	execCmd(cmd)
	
def RunBuild():
	global KernelType
	
	if(os.path.exists("./out")):
		shutil.rmtree("./out")
	os.mkdir("./out")
	os.chdir("./out")

	if(len(sys.argv) > 1):
		#if(sys.argv[1] == "asm"):
		#	KernelType = "ASM_KERNEL"
		if(sys.argv[1] == 'hip'):
			KernelType = "HIP_KERNEL"
			
	BuildTarget()
	RunTarget()	

if __name__ == '__main__':
	RunBuild()
	exit()	
