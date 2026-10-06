import json
from pathlib import Path
from sdcpp_engine import check_execution_options,conditioning_hits
ref=json.loads(Path('configs/a100-sdcpp-flux-klein-base-cache-control.resolved.json').read_text())
for suffix in ['', '-smoke']:
 c=json.loads(Path('configs/a100-sdcpp-flux-klein-base-conditioning-cache'+suffix+'.resolved.json').read_text())
 for k in ['model','precision','workload','engine']:assert c[k]==ref[k]
 check_execution_options(c)
 assert conditioning_hits(c,2)==[0,2]
 assert {k:v for k,v in c['engine_settings'].items() if ref['engine_settings'][k]!=v}=={'conditioning_cache_size':2}
 print(suffix or 'full','validated')
