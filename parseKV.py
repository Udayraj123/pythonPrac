import sys
import pandas as pd
import os
from os import listdir

def find_xlsx_filenames( path_to_dir, suffix=".xlsx" ):
    filenames = sorted(listdir(path_to_dir))
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

skipline=9
directory= os.getcwd()
output_dir=directory+'/parsed';
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = ["Technothlon '17 Registration Sheet.xlsx"];
args=sys.argv
if(len(args)>1):
	if(args[1]!='.'):
		directory = directory+'/'+args[1]
	files = find_xlsx_filenames(directory)
	if(len(args)>2):
		skipline=int(args[2])

def isin(subs,string):
	for sub in subs:
		if sub in string:
			return True
	return False

def verify(cols):
	verify= [['s. no','s.no','no','id'], ['squad(junior/hauts)','squad'], ['medium(english/hindi)','medium','language'], ['name 1','name'], ['name 2','name'], ['e-mail','e mail','email'], ['e-mail','e mail','email'], ['contact1','conatct1','contact-1','contact'],['contact2','conatct2','contact-2','contact']];
	if(len(cols)!=len(verify)):
		return False;
	for i,col in enumerate(cols):
		if(not isin(verify[i],str(col).lower())):
			print("Mismatch for "+str(col).lower()+" in ",verify[i])

			return False
	return True

	
headers = ['SN','squad', 'language','name1', 'name2', 'email1', 'email2', 'contact1', 'contact2' ]
regHeaders= ['name1', 'name2', 'email1', 'email2', 'contact1', 'contact2', 'squad', 'language']
newheaders=['filename','count','School Name','address','email','contact','fac_name','fac_email','fac_contact']
schooldetails = []
for filename in files:
	print("Converting "+filename)
	t = pd.read_excel(directory+'/'+filename,parse_cols = len(headers)-1,skiprows=skipline)
	num_rows = t.shape[0] - 1  #exclude header
	t2 = pd.read_excel(directory+'/'+filename,header=None,skiprows=1,skip_footer=num_rows+3,parse_cols=3,names=['field','d1','d2','values'])
	if(not verify(list(t.columns))):
		print("Skipping file, Wrong format for " + filename);
		continue
	t.columns=headers
	t = t[regHeaders] #rearrange the columns
	# t=t[skipline:] # skip princi name etc
	t = t.dropna(how='all') #drop only if all values NA i.e. blank rows
	regCount = t.shape[0]
	t.to_csv(output_dir+'/'+ filename.split('.xlsx')[0] +'_'+str(regCount)+'.csv',index=False) 
	
	# for col in regHeaders[4:]:
	# 	t[col] = t[col].apply(lambda r: r if (str(r).lower()!='principal') else "")

	if(t2.shape[0]==7):
		schooldetails.append([filename,regCount]+list(t2['values']))
	else:
		print("School Details Invalid")
		print(t2)

x = pd.DataFrame(schooldetails,columns=newheaders)
x.to_csv(output_dir+'/SchoolList_.csv',index=False)
print("Files are stored at " + output_dir)