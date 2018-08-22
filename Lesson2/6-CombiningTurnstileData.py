#reference
#https://gist.github.com/Mengyuz/6a172044d96cd0998ce7
def create_master_turnstile_file(filenames, output_file):
    with open(output_file, 'w') as master_file:
        master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
        for name in filenames:
            #open each name
            with open(name, 'r') as read_data:
                for line in read_data:
                    master_file.write(line)#write each line
