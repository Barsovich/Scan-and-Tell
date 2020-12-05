import pexpect
import subprocess

def download_filetype(filetype):
    child = pexpect.spawn('python2 download-scannet.py -o ../data/scannet --type ' + filetype)
    child.expect('Press any key to continue, or CTRL-C to exit.', timeout=30000)
    child.sendline('y')    
    child.expect('Press any key to continue, or CTRL-C to exit.', timeout=30000)
    child.sendline('y')

process = subprocess.Popen(['wget', 'https://raw.githubusercontent.com/xjwang-cs/TSDF_utils/master/download-scannet.py'], stdin=subprocess.PIPE)
process.wait()
download_filetype('_vh_clean_2.ply')
download_filetype('_vh_clean.aggregation.json')
download_filetype('_vh_clean_2.0.010000.segs.json')
download_filetype('.txt')
download_filetype('.aggregation.json')