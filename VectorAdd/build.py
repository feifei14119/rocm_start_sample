import subprocess
import sys, os, re
import shutil

Target = "vector_add.out"

KernelType = "-D \"ASM_KERNEL\" "
ObjectVersion = "-D \"OBJ_V2\" "
Compiler = "-D \"CMP_HCC\" "

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
		
	cmd = 'hipcc ../main.cpp ../../common/utils.cpp ' + KernelType + ObjectVersion + Compiler + '-O0 -w -std=c++11 -o ' + Target
	print(cmd)
	text = execCmd(cmd)
	print(text)
	
	return
	
def RunTarget():
	# remove kernel bin
	if os.path.exists("./*.bin"):
		os.remove("./*.bin")
	if os.path.exists("./*.o"):
		os.remove("./*.o")
	
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
	
	if(len(sys.argv) > 1):
		if(sys.argv[1] == "asm"):
			KernelType = "-D \"ASM_KERNEL\" "
		if(sys.argv[1] == "hip"):
			KernelType = "-D \"HIP_KERNEL\" "
	if(len(sys.argv) > 2):
		if(sys.argv[2] == "v2"):
			ObjectVersion = "-D \"OBJ_V2\" "
		if(sys.argv[2] == "v3"):
			ObjectVersion = "-D \"OBJ_V3\" "
	if(len(sys.argv) > 3):
		if(sys.argv[3] == "hcc"):
			Compiler = "-D \"CMP_HCC\" "
		if(sys.argv[3] == "llvm"):
			Compiler = "-D \"CMP_LLVM\" "
			
	BuildTarget()
	RunTarget()	

if __name__ == '__main__':
	RunBuild()
	exit()
	
	
