#!/usr/bin/env python3
"""Generate captions with preserved, sanitized prompt templates and ms-swift."""
import argparse, json
from pathlib import Path
TEMPLATE_FILE=Path(__file__).with_name("sagittal_prompt_templates.json")

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--input",required=True,help="UTF-8 JSONL: template_id, input_data, optional request_id")
 p.add_argument("--output",required=True)
 p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8000)
 p.add_argument("--batch-size",type=int,default=20); p.add_argument("--max-tokens",type=int,default=512)
 p.add_argument("--temperature",type=float,default=1.0)
 g=p.add_mutually_exclusive_group(); g.add_argument("--append",action="store_true")
 g.add_argument("--overwrite",action="store_true")
 p.add_argument("--validate-only",action="store_true")
 a=p.parse_args()
 if a.batch_size<1 or a.max_tokens<1: p.error("batch-size and max-tokens must be positive")
 templates=json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
 by_id={x["template_id"]:x["prompt"] for x in templates}
 rows=[]
 for n,line in enumerate(Path(a.input).read_text(encoding="utf-8").splitlines(),1):
  if not line.strip(): continue
  try: row=json.loads(line)
  except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON on line {n}: {e.msg}") from e
  if not isinstance(row,dict) or not isinstance(row.get("template_id"),str) or not isinstance(row.get("input_data"),str):
   raise ValueError(f"Line {n} requires string fields 'template_id' and 'input_data'")
  if row["template_id"] not in by_id: raise ValueError(f"Line {n}: unknown template_id {row['template_id']!r}")
  if "request_id" in row and not isinstance(row["request_id"],(str,int)):
   raise ValueError(f"Line {n}: request_id must be a string or integer")
  prompt=by_id[row["template_id"]].replace("{{INPUT_DATA}}",row["input_data"])
  rows.append((row,prompt))
 if not rows: raise ValueError("Input contains no requests")
 out=Path(a.output)
 if out.exists() and not (a.append or a.overwrite): raise FileExistsError(f"{out} exists; pass --append or --overwrite")
 if a.validate_only:
  print(f"Validated {len(rows)} prompt requests; inference not run."); return
 from swift.llm import InferClient, InferRequest, RequestConfig
 try:
  import nest_asyncio; nest_asyncio.apply()
 except ImportError: pass
 client=InferClient(host=a.host,port=a.port)
 config=RequestConfig(max_tokens=a.max_tokens,temperature=a.temperature)
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("a" if a.append else "w",encoding="utf-8") as f:
  for start in range(0,len(rows),a.batch_size):
   batch=rows[start:start+a.batch_size]
   requests=[InferRequest(messages=[{"role":"user","content":prompt}]) for _,prompt in batch]
   responses=client.infer(requests,config)
   if len(responses)!=len(batch): raise RuntimeError(f"Expected {len(batch)} responses, got {len(responses)}")
   for i,((row,_),response) in enumerate(zip(batch,responses),start):
    result={"index":i,"template_id":row["template_id"],"caption":response.choices[0].message.content}
    if "request_id" in row: result["request_id"]=row["request_id"]
    f.write(json.dumps(result,ensure_ascii=False)+"\n")
   f.flush()
 print(f"Generated {len(rows)} captions: {out}")
if __name__=="__main__": main()
