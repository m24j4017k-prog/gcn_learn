import yaml
from Processor import Processor

from utils.arg_parser import get_parser
from utils.seed import init_seed


        
    
if __name__ == '__main__':
    
    # コマンドライン引数での設定の読み込み
    parser = get_parser()
    p = parser.parse_args()
    
    # コンフィグファイルでの設定の読み込み 
    file_setting = {}
    if p.config is not None:
        with open(p.config, 'r') as f:
            file_setting = yaml.safe_load(f)
            
        cmd_key = vars(p).keys()
        
        # get_parser関数で定義されていない設定内容がコンフィグファイルで書かれていた場合はエラーを出す     
        for file_k in file_setting.keys():
            if file_k not in cmd_key:
                print(
                    f'{file_k} が定義されていない設定内容です\n'
                    f'設定内容が間違っているか、config/arg_parser.pyに設定内容を追加してください'
                )
                assert(file_k in cmd_key)
    
    # 設定ファイルを読み込む
    parser.set_defaults(**file_setting)
    arg = parser.parse_args()
    
    # ランダムシードを固定
    init_seed(0)
    
    
    processor = Processor(arg)
    processor.train()

