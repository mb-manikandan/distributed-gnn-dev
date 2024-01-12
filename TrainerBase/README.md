Samplerベースの分散GNN(DistGNN)のフレームワーク化
=======

SALIENTなどのサンプラーベースの手法(Distributed Mini-batch TrainingのIndividual-sample-based ExecutionまたはJoint-sample-based Execution)から
DistGNNの代表的な実装方法を調査し、簡単なサンプル(スケルトン化したソース)の実装をする。

## 修正方法とソースの仕分けについて
* SALIENTソースをコピーし、徐々に不要な箇所を削除していく。 後にディレクトリ構造も整理していく。
* SALIENTなどの特徴（FastSamplerなど）は省き、DistGNNの特徴のみ抽出し、なるべく最低限のスケルトン(骨組み)のみを抽出する。
* SALIENT固有のソースは"SALIENT_tools"に保管する。また汎用性のある便利機能は"Common_tools"などに仕分けることにする。
* 各機能の詳細は実装のタイミングで読む。
* 今後の汎かを考慮して、汎用性の高いソースへ修正する(DDP通信処理を引数渡しで実行できるようにするなど)。

## DistGNNの必要箇所
* Slurmで計算機ごと、torchのmp.sparn関数でプロセス(デバイス)ごとに、プロセスを分散させている。
また、torchのDDP処理でデータ分散させている。ddp.pyのDDPDriverクラスでDDP通信処理も実装している。
* サンプラー(PyGのNeighborSamplerなど)をbase.py,ddp.pyで定義と初期化し、train.pyのserial_train関数内でサンプリング処理を行っている。

※ログ機能はBaseDriverに残す。

## Samplerベースの分散GNNのフレームワーク化の主要クラス
### outer_frame/simple_frame.py SimpleFrameクラス
* 外枠のフレームワーククラス。main.pyの汎用箇所(Epochループ処理などのディープラーニングの主な処理)を1つのクラスにまとめた。
* SimpleFrameクラスはそのまま利用が可能である。使い方はexample/ns_main.pyを参考にする。
* 必要に応じてオプション付きの機能追加も可能になるように、クラス継承できるように構成した
(参考；outer_frame/optional_frame.py OptionalFrameクラス)。

### inner_frame/sampler_base.py SamplerBaseDriver2クラス
* 内枠のフレームワーククラス。base.pyとddp.pyのDriverクラスの汎用箇所(DDP通信処理やsamplerの空定義、学習処理のフレームワークなど)を1つのクラスにまとめた。
* SamplerBaseDriver2クラスは抽象クラスなので、このクラスを継承してSampler定義と学習,テスト処理を実装(定義)したクラスを用意しなければいけない。
使い方はexample/ns_driver.pyを参考にする。


## Samplerベースの分散GNNのフレームワークを利用したPyG NeighborSamplerのExampleプログラム
以下にexample直下にPyG NeighborSamplerによる参考プログラムを配置。
* ns_driver.py: SamplerBaseDriver2クラスを継承したクラスをNSDriverクラスを定義している。
PyG NeighborSamplerの定義や下記ns_trainerの学習やテスト処理を配置している。
* ns_trainer: PyG NeighborSamplerを利用した学習やテスト処理を定義している。
* ns_main.py: SimpleFrameクラスとNSDriverクラスを使用し、よりシンプルな分散GNNを実行するメインプログラムである。


## SALIENTの特徴箇所
以下はSALIENT_toolsファルダに移動させる。
* FastSampler(fast_sampler,fast_trainer/sampler.pyなど)
* シングルプロセス起動クラス(driver/drivers/singleproc.py)


## 汎用性のある便利機能の箇所
以下はCommon_toolsファルダに移動させる。
* CUDAイベント時刻取得関数とRuntime Statics機能(utils.py)
* timer機能(utils.py)
* プログレスバー機能(progres_bar.py)
* 学習率調整関数(lr_scheduler.py)



