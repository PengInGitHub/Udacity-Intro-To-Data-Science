import csv

#resource
#https://github.com/tuanavu/udacity-course/blob/master/introduction_to_data_science/lesson_2/exercise/5%20-%20Fixing%20Turnstile%20Data.ipynb
#https://gist.github.com/Mengyuz/a2e0413037ac73675029
def fix_turnstile_data(filenames):
    '''
    Filenames is a list of MTA Subway turnstile text files. A link to an example
    MTA Subway turnstile text file can be seen at the URL below:
    http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt
    
    As you can see, there are numerous data points included in each row of the
    a MTA Subway turnstile text file. 

    You want to write a function that will update each row in the text
    file so there is only one entry per row. A few examples below:
    A002,R051,02-00-00,05-28-11,00:00:00,REGULAR,003178521,001100739
    A002,R051,02-00-00,05-28-11,04:00:00,REGULAR,003178541,001100746
    A002,R051,02-00-00,05-28-11,08:00:00,REGULAR,003178559,001100775
    
    Write the updates to a different text file in the format of "updated_" + filename.
    For example:
        1) if you read in a text file called "turnstile_110521.txt"
        2) you should write the updated data to "updated_turnstile_110521.txt"

    The order of the fields should be preserved. Remember to read through the 
    Instructor Notes below for more details on the task. 
    
    In addition, here is a CSV reader/writer introductory tutorial:
    http://goo.gl/HBbvyy
    
    You can see a sample of the turnstile text file that's passed into this function
    and the the corresponding updated file by downloading these files from the resources:
    
    Sample input file: turnstile_110528.txt
    Sample updated file: solution_turnstile_110528.txt
    '''
    for name in filenames:
        with open(name, 'rb') as f:#open file
            reader = csv.reader(f) #csv reader
            with open('updated_'+name, 'wb') as f: #write file
                writer = csv.writer(f) #csv writer
                updated_lines = []
                for line in reader:
                    updated_line = line[0:3]#get first 3 element
                    #header: every first 3 ele in each line
                    #append ele to this header
                    for i in range(0,len(line)-3):#iterate the rest ele in this line
                        updated_line.append(line[i+3])#append next ele
                        if (i+1)%5 == 0: #every five ele added to the header
                            updated_lines.append(updated_line) #add this composed content(3+5) into result
                            updated_line = line[0:3]#refresh
                writer.writerows(updated_lines)



