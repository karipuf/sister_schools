import pandas as pd,pylab as pl,numpy as np,sys,pickle,io,lightgbm as lgb,pyspark,openai,base64,os,json
from json_repair import repair_json

GEMINI_FLASH_MODEL='gemini-2.5-flash-preview-04-17'
GEMINI_PRO_MODEL='gemini-2.5-pro-preview-03-25'

def match_images(url1,url2,client,model='gemma3:12b',debug=False):
    
     imdat1=base64.b64encode(open(url1,"rb").read()).decode("utf-8")
     imdat2=base64.b64encode(open(url2,"rb").read()).decode("utf-8")

     the_prompt=f"""
Please study these two images, and let me know if the two individuals shown here are holding the same object.
Please respond in json format as follows: {{'image1':'{url1}','image2':'{url2}','matching':<true/false>}}
Please only return 'matching' as True if the two objects are very clearly the same thing.
  
Please respond ONLY with the json object, no need for any extra explanation or comments.
"""
     if debug: print(the_prompt)
     
     try:
         res=client.chat.completions.create(model=model,messages=[{'role':'user','content':[{'type':'text','text':the_prompt},
                                                    {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat1}"}},
                                                    {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat2}"}}
                                                    ]}])
         
         return json.loads(repair_json(res.choices[0].message.content))

     except:

         return {'image1':url1,'image2':url2,'matching':'error'}
