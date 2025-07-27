# rpgmaker-translator
gemini 自动汉化 rpgmaker mv 游戏脚本
## 缘起
有一个想玩的黄油，然而找到的资源都是依赖 mtool 进行机翻汉化，不仅对我这种 macOS 玩家不友好不说，机翻的质量也是一眼难尽...

虽然我懒得写代码，但是指挥一下ai写代码的闲心还是有的，ai 相当适合写这种不复杂的脚本。

## 使用

1. cd 到游戏目录下的 data 目录
2. 运行该脚本 (请自行安装依赖)

> python ./translator.py --api-key <api-key> --api-url <api-url> --model gemini-2.0-flash

api-key 和 api-url 默认是用的 closeai 的代理，推荐使用，充个10块钱差不多就够汉化一部游戏了
