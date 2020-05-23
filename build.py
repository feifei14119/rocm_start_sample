import subprocess
import sys, os, re
import shutil

def execCmd(cmd):  
	r = os.popen(cmd)  
	text = r.read() 
	print(text)
	r.close()  
	return text 

def buildAsmApp():
	if os.path.exists("../rocm_start_sample.out"):
		print("remove ../rocm_start_sample.out")
		os.remove("../rocm_start_sample.out")
		
	cmd = 'hipcc ../src/main.cpp -c -fPIC -D"ASM_KERNEL" -O0 -w -std=c++11 '
	print(cmd)
	execCmd(cmd)
	
	cmd = 'g++ -o ./rocm_start_sample.out ./main.o /opt/rocm/lib/libhip_hcc.so'
	print(cmd)
	execCmd(cmd)
	
def buildHipApp():
	if os.path.exists("../rocm_start_sample.out"):
		print("remove ../rocm_start_sample.out")
		os.remove("../rocm_start_sample.out")
		
	cmd = 'hipcc ../src/main.cpp -c -fPIC -D"HIP_KERNEL" -O0 -w -std=c++11 '
	print(cmd)
	execCmd(cmd)
	
	cmd = 'g++ -o ./rocm_start_sample.out ./main.o /opt/rocm/lib/libhip_hcc.so'
	print(cmd)
	execCmd(cmd)
	
def runApp():	
	cmd = './rocm_start_sample.out'
	print(cmd)
	execCmd(cmd)
	
if __name__ == '__main__':
	if(os.path.exists("./out")):
		shutil.rmtree("./out")
	os.mkdir("./out")
	os.chdir("./out")

	if(len(sys.argv) == 1):
		buildAsmApp()
		runApp()
		exit()
	
	arg = sys.argv[1]
	
	if(arg == "asm"):
		buildAsmApp()
		runApp()
		exit()
	
	if(arg == 'hip'):
		buildHipApp()
		runApp()
		exit()
	
	
