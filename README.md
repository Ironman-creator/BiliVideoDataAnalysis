# BiliVideoDataAnalysis
本项目重点分析Bilibili排名视频信息。它执行全面的数据处理，包括数据清理和预处理，以确保数据完整性。通过各种数据分析和可视化技术，如相关性分析、发布位置和视频类别的分布分析、交互类型分析以及使用随机森林模型的特征重要性分析，它揭示了关于Bilibili视频的宝贵见解。此外，基于视频标题生成一个词云来展示流行关键字。该项目在Pandas、Matplotlib、Seaborn、Scikit-learn和WordCloud等库的帮助下用Python实现，旨在深入了解Bilibili视频生态系统和相关数据特征。
# 一、项目概述
本项目旨在对 B 站排行榜视频信息进行全面的数据分析与可视化，深入挖掘数据背后的规律与特征，以更好地理解 B 站视频生态的相关情况。通过数据处理、多种统计分析和丰富的可视化手段，探索视频的各项属性，如发布位置、分类、互动数据之间的关系，以及不同特征对点赞数的影响等，并绘制了标题词云图以直观呈现视频标题的热点词汇分布。
# 二、功能特点
1.数据预处理：  
读取 B 站排行榜视频信息的 Excel 文件，并对数据进行完整性检查，包括查看数据基本信息、统计缺失值和处理重复值，确保数据的质量和可用性。  
对缺失的作者和发布位置信息进行填充，将其替换为 “暂无”，使数据更加完整。  
2.数据分析与可视化：  
相关性分析：计算数值型指标（如播放数、弹幕数等）的相关系数矩阵，并绘制热力图，清晰展示各指标之间的相关性强弱。  
发布位置分析：统计每个发布位置的视频数量，选取排名前十的发布位置绘制水平条形图，直观呈现不同发布位置的视频分布情况。  
分类分析：按照播放数对视频分类进行排序，选取前 20 个分类绘制条形图，揭示不同分类在播放量方面的差异。  
互动类型分析：对点赞数、弹幕数、回复数、收藏数、投币数和分享数等互动指标进行统计，绘制饼状图展示各互动类型在整体中的占比，反映观众的互动行为分布。  
特征重要性分析：通过构建随机森林回归模型，评估播放数、弹幕数、回复数等多个特征对点赞数的贡献度，选取排名前十的特征绘制特征重要性图，帮助理解影响点赞数的关键因素。  
词云图绘制：将所有视频标题合并为一个文本，使用指定的中文字体生成词云图，突出显示标题中的高频词汇，从而了解视频主题的热点趋势。  
# 三、技术栈
Python：作为主要的编程语言进行数据处理和分析。  
Pandas：用于数据读取、清洗和预处理，提供高效的数据操作和分析功能。  
Matplotlib：强大的绘图库，实现各种统计图表的绘制，如条形图、饼状图、热力图等。  
Seaborn：基于 Matplotlib 的高级数据可视化库，使绘制美观且信息丰富的统计图形更加便捷，如相关性热力图。  
Scikit-learn：用于构建随机森林模型进行特征重要性分析，提供了丰富的机器学习算法和工具。  
WordCloud：用于生成标题词云图，直观展示文本数据中的关键词分布。  
# 四、使用方法
1.确保已经安装了上述所需的 Python 库。  
2.将 B站排行榜视频信息.xlsx 文件放置在项目目录下。  
3.运行代码，代码将依次执行数据预处理、各项数据分析与可视化以及词云图绘制操作。在运行过程中，会生成一系列的图表展示数据的分析结果，包括热力图、条形图、饼状图、特征重要性图和词云图等，可直观查看和分析 B 站视频数据的相关特征和规律。
# 五、注意事项
1.运行代码前，请检查 B站排行榜视频信息.xlsx 文件的路径和名称是否正确，确保数据能够正常读取。  
2.若在运行过程中出现与库版本相关的问题（如某些属性或方法不存在），可能需要根据实际情况升级或调整相关库的版本，例如对于 scikit-learn 库中 OneHotEncoder 的不同版本兼容性问题，可参考代码中的注释进行处理。  
3.词云图绘制时指定的中文字体路径需根据实际情况进行调整，如果字体文件不存在或路径错误，可能导致词云图中的中文无法正确显示。  
