import time

localtime = time.asctime( time.localtime(time.time()) )
print ("完成时间为 :", localtime)