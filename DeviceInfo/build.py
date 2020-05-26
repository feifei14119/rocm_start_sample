import subprocess
import sys, os, re
import shutil

Target = "device_info.out"

def execCmd(cmd):		
	r = os.popen(cmd)  
	text = r.read() 
	print(text)
	r.close()  
	return text 
	
def BuildTarget():
	if os.path.exists("./" + Target):
		os.remove("./" + Target)
		
	cmd = 'hipcc ../main.cpp -O0 -w -std=c++11 -o ' + Target
	print(cmd)
	text = execCmd(cmd)
	print(text)
	
	return
	
def RunTarget():	
	cmd = "./" + Target
	print(cmd)
	execCmd(cmd)
	
if __name__ == '__main__':	
	if(os.path.exists("./out")):
		shutil.rmtree("./out")
	os.mkdir("./out")
	os.chdir("./out")
			
	BuildTarget()
	RunTarget()
	exit()
	
	
