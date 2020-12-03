def exec_and_output(command):
  p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  print(output.decode("utf-8") )
  print("Return code: ", p_status )