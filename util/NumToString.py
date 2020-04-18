import sys,re

def ReadDic(dic_file):
    open_dic=open(dic_file,"r")
    dic={}
    index=0
    for item in open_dic:
        item=item.strip()
        index=index+1
        dic[str(index)]=item
    return dic
    
def Write(inputfile,outputfile,Dic):
    input=open(inputfile,"r")
    output=open(outputfile,"w")
    for line in input:
        G=re.split(" |\t",line.lower().strip())
        for index, item in enumerate(G):  
            if index == 0:
                output.write(str(item) + " ") # target class
                continue
            if item=="":
                continue
            if item in Dic:
                output.write(str(Dic[item])+" ")
            else:
                output.write("1 ")
        output.write("\n")
    input.close()
    output.close()


dic=ReadDic(sys.argv[1])
Write(sys.argv[2],sys.argv[3],dic)
