"""Local environment diagnosis. Does not connect or send orders."""
from pathlib import Path
import importlib.util
import os
import platform
import sys
from .ops_v9 import atomic_json

def environment_report(base,settings,data_directory,now,audit):
    base=Path(base)
    def installed(module):
        try:return importlib.util.find_spec(module) is not None
        except (ImportError,ValueError):return False
    has_token=os.getenv('TUSHARE_TOKEN','').strip() not in {'','put_your_token_here'}
    envfile=base/'.env'
    if not has_token and envfile.exists():
        for line in envfile.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k,v=line.split('=',1)
                if k.strip()=='TUSHARE_TOKEN':has_token=bool(v.strip().strip('"').strip("'") not in {'','put_your_token_here'})
    modules={m:installed(m) for m in ['pandas','numpy','scipy','tushare','xtquant']}
    current=audit['status']=='PASS';core=all(modules[m] for m in ['pandas','numpy','scipy'])
    next_steps=[]
    if not core:next_steps.append('按 README 安装 requirements-live.txt')
    if not modules['tushare'] or not has_token:next_steps.append('下载数据前安装 tushare，并在本机 .env 设置自己的 TUSHARE_TOKEN')
    if not current:next_steps.append('运行 python live.py prepare --download；数据权限或一致性错误需按报告修复')
    else:next_steps.append('运行 python live.py prepare；盘中刷新券商账户和行情后再生成计划')
    next_steps.append('未完成你的券商环境联调和真实策略验证；本诊断不是上线许可')
    return {'schema_version':9,'checked_at':now.isoformat(),'python':sys.version.split()[0],
      'platform':platform.system(),'modules':modules,'tushare_token_present':has_token,
      'download_environment_ready':core and modules['tushare'] and has_token,
      'signal_preparation_ready':core and current,'data_audit':audit,
      'qmt_local_sdk_present':platform.system()=='Windows' and modules['xtquant'],
      'live_enabled':settings.get('live_enabled') is True,'broker_rules_confirmed':settings.get('broker_rules_confirmed') is True,
      'broker_connection_tested':False,'performance_validated':False,'next_steps':next_steps}

def write_doctor_report(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);atomic_json(directory/'doctor.json',report)
    rows=['# 运行环境检查','',f"检查时间：{report['checked_at']}",'',
      '| 项目 | 结果 |','|---|---|',f"| 系统 / Python | {report['platform']} / {report['python']} |",
      f"| 下载环境 | {'具备本地依赖和Token，未测试远程权限' if report['download_environment_ready'] else '尚未配齐'} |",
      f"| 数据一致性 | {report['data_audit']['status']} |",
      f"| 真实下单开关 | {'已开启，请核对本机配置' if report['live_enabled'] else '关闭'} |",
      '| 券商连接与策略业绩 | 未验证 |','','## 下一步','']
    rows += [f'{i}. {text}' for i,text in enumerate(report['next_steps'],1)]
    rows += ['','## 数据检查详情','']+[f"- {r['check']}：{r['status']} — {r['detail']}" for r in report['data_audit']['checks']]
    (directory/'doctor.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
