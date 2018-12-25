# elo
Kaggle Elo Competition

## Insights
* Targetとpurchase_amountはほぼ相関していないが、purchase_amountのsumやmaxが大きいほど、Targetのバラツキは小さくなっている。


## Experience
### 113 12/25
Targetにおける-33の外れ値の除去と絶対値6~9を超える値の収縮を行って学習を行い、外れ値の予測値のみ  
1. classfierで0.1以上の予測値を持つIDを外れ値にする  
2. 過去最高submitにて-6~-3を超える予測値になっているIDのみ、過去最高submitの予測値に置き換える
  
上記2つの方法で予測値を補強した。しかし、1.はLB4.3, 2.はLB3.76~3.80という結果だった。（4回submit消費, 現状のベストLBは0.3697）  
外れ値以外の予測は外れ値を除いた方が精度が上がると考えられるので、何か見落としている。

### 
