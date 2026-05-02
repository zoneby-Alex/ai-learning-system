import { defineConfig } from 'vitepress'

// 侧边栏配置 - 映射整个知识体系
const sidebar = {
  '/0-学习路线图/': [
    { text: '学习路线图总览', link: '/0-学习路线图/' },
    { text: '3个月速通版', link: '/0-学习路线图/3个月速通版' },
    { text: '6个月就业版', link: '/0-学习路线图/6个月就业版' },
    { text: '12个月深入版', link: '/0-学习路线图/12个月深入版' },
  ],
  '/1-数学基础/': [
    { text: '数学基础总览', link: '/1-数学基础/' },
    {
      text: '线性代数',
      collapsed: false,
      items: [
        { text: '向量与矩阵运算', link: '/1-数学基础/线性代数/知识点-向量与矩阵运算' },
        { text: '范数', link: '/1-数学基础/线性代数/知识点-范数' },
        { text: '特征值与SVD', link: '/1-数学基础/线性代数/知识点-特征值与SVD' },
        { text: '矩阵微分', link: '/1-数学基础/线性代数/知识点-矩阵微分' },
      ]
    },
    {
      text: '微积分',
      collapsed: true,
      items: [
        { text: '导数与梯度', link: '/1-数学基础/微积分/知识点-导数与梯度' },
        { text: '泰勒展开', link: '/1-数学基础/微积分/知识点-泰勒展开' },
      ]
    },
    {
      text: '概率统计',
      collapsed: true,
      items: [
        { text: '概率分布', link: '/1-数学基础/概率统计/知识点-概率分布' },
        { text: '贝叶斯公式', link: '/1-数学基础/概率统计/知识点-贝叶斯公式' },
        { text: '最大似然估计', link: '/1-数学基础/概率统计/知识点-最大似然估计' },
        { text: '期望方差协方差', link: '/1-数学基础/概率统计/知识点-期望方差协方差' },
      ]
    },
    {
      text: '信息论',
      collapsed: true,
      items: [
        { text: '熵', link: '/1-数学基础/信息论/知识点-熵' },
        { text: '交叉熵', link: '/1-数学基础/信息论/知识点-交叉熵' },
        { text: 'KL散度', link: '/1-数学基础/信息论/知识点-KL散度' },
      ]
    },
    {
      text: '最优化方法',
      collapsed: true,
      items: [
        { text: '梯度下降', link: '/1-数学基础/最优化/知识点-梯度下降' },
        { text: 'Adam优化器', link: '/1-数学基础/最优化/知识点-Adam优化器' },
        { text: '学习率调度', link: '/1-数学基础/最优化/知识点-学习率调度' },
        { text: '梯度裁剪', link: '/1-数学基础/最优化/知识点-梯度裁剪' },
        { text: '拉格朗日乘子', link: '/1-数学基础/最优化/知识点-拉格朗日乘子' },
      ]
    },
  ],
  '/2-机器学习/': [
    { text: '机器学习总览', link: '/2-机器学习/' },
    {
      text: '监督学习',
      collapsed: false,
      items: [
        { text: '线性回归', link: '/2-机器学习/监督学习/知识点-线性回归' },
        { text: '逻辑回归', link: '/2-机器学习/监督学习/知识点-逻辑回归' },
        { text: 'SVM', link: '/2-机器学习/监督学习/知识点-SVM' },
        { text: '决策树', link: '/2-机器学习/监督学习/知识点-决策树' },
      ]
    },
    {
      text: '无监督学习',
      collapsed: true,
      items: [
        { text: 'KMeans', link: '/2-机器学习/无监督学习/知识点-KMeans' },
        { text: 'PCA', link: '/2-机器学习/无监督学习/知识点-PCA' },
      ]
    },
    {
      text: '集成学习',
      collapsed: true,
      items: [
        { text: 'XGBoost', link: '/2-机器学习/集成学习/知识点-XGBoost' },
        { text: 'LightGBM', link: '/2-机器学习/集成学习/知识点-LightGBM' },
        { text: 'Bagging与Boosting', link: '/2-机器学习/集成学习/知识点-Bagging与Boosting' },
      ]
    },
  ],
  '/3-深度学习核心/': [
    { text: '深度学习核心总览', link: '/3-深度学习核心/' },
    {
      text: '基础原理',
      collapsed: false,
      items: [
        { text: 'BP反向传播', link: '/3-深度学习核心/BP反向传播/README' },
        { text: '自动微分', link: '/3-深度学习核心/BP反向传播/知识点-自动微分' },
        { text: '激活函数', link: '/3-深度学习核心/激活函数/README' },
        { text: '初始化策略', link: '/3-深度学习核心/初始化/知识点-初始化策略' },
        { text: '正则化', link: '/3-深度学习核心/正则化/README' },
      ]
    },
    {
      text: 'CNN / RNN',
      collapsed: true,
      items: [
        { text: 'CNN家族', link: '/3-深度学习核心/CNN家族/README' },
        { text: 'RNN家族', link: '/3-深度学习核心/RNN家族/README' },
      ]
    },
    {
      text: 'Transformer',
      collapsed: true,
      items: [
        { text: 'Self-Attention', link: '/3-深度学习核心/Transformer/知识点-Self-Attention' },
        { text: 'ViT多模态', link: '/3-深度学习核心/ViT多模态/README' },
      ]
    },
    {
      text: '进阶主题',
      collapsed: true,
      items: [
        { text: 'Diffusion', link: '/3-深度学习核心/Diffusion/README' },
        { text: 'MoE架构', link: '/3-深度学习核心/MoE/知识点-MoE架构' },
        { text: 'LoRA微调', link: '/3-深度学习核心/LoRA/知识点-LoRA微调' },
        { text: 'RLHF与DPO', link: '/3-深度学习核心/训练技术/知识点-RLHF与DPO' },
      ]
    },
  ],
  '/4-LLM与生成式AI/': [
    { text: 'LLM模块总览', link: '/4-LLM与生成式AI/' },
    { text: 'LLM必读论文清单', link: '/4-LLM与生成式AI/LLM必读论文清单' },
    {
      text: '模型系列',
      collapsed: false,
      items: [
        { text: 'GPT系列演化', link: '/4-LLM与生成式AI/GPT系列/知识点-GPT演化' },
        { text: 'Llama系列演化', link: '/4-LLM与生成式AI/Llama系列/知识点-Llama演化' },
        { text: '国产模型对比', link: '/4-LLM与生成式AI/国内模型/知识点-国产模型对比' },
      ]
    },
    {
      text: '训练与对齐',
      collapsed: true,
      items: [
        { text: 'Pretrain训练', link: '/4-LLM与生成式AI/模型训练/知识点-Pretrain训练' },
        { text: 'SFT与对齐', link: '/4-LLM与生成式AI/模型训练/知识点-SFT与对齐' },
      ]
    },
    {
      text: '应用技术',
      collapsed: true,
      items: [
        { text: 'RAG核心原理', link: '/4-LLM与生成式AI/RAG系统/知识点-RAG核心原理' },
        { text: '推理优化技术', link: '/4-LLM与生成式AI/推理优化/知识点-推理优化技术' },
      ]
    },
  ],
  '/5-前沿方向/': [
    { text: '前沿方向总览', link: '/5-前沿方向/' },
    { text: 'AI Agent', link: '/5-前沿方向/AI-Agent/README' },
    { text: 'AI Infra / 推理优化', link: '/5-前沿方向/AI-Infra/知识点-推理优化' },
    { text: '多模态', link: '/5-前沿方向/多模态/README' },
    { text: 'AI Coding', link: '/5-前沿方向/AI-Coding/README' },
    { text: '长上下文', link: '/5-前沿方向/长上下文/README' },
    { text: 'AI4Science', link: '/5-前沿方向/AI4Science/知识点-AI4Science应用' },
    { text: '具身智能', link: '/5-前沿方向/Embodied-AI/知识点-具身智能' },
  ],
  '/6-项目实战/': [
    { text: '项目实战总览', link: '/6-项目实战/' },
    {
      text: '初级项目',
      collapsed: false,
      items: [
        { text: 'MNIST分类器', link: '/6-项目实战/初级项目/项目-MNIST分类器' },
        { text: 'CIFAR-10图像分类', link: '/6-项目实战/初级项目/项目-CIFAR10图像分类' },
        { text: '文本分类(IMDB)', link: '/6-项目实战/初级项目/项目-文本分类IMDB' },
      ]
    },
    {
      text: '中级项目',
      collapsed: true,
      items: [
        { text: 'ViT实现', link: '/6-项目实战/中级项目/项目-ViT实现' },
        { text: 'Diffusion复现', link: '/6-项目实战/中级项目/项目-Diffusion复现' },
        { text: 'RAG知识库系统', link: '/6-项目实战/中级项目/项目-RAG知识库系统' },
      ]
    },
    {
      text: '高级项目',
      collapsed: true,
      items: [
        { text: '多模态Agent系统', link: '/6-项目实战/高级项目/项目-多模态Agent系统' },
        { text: 'LLM微调实践', link: '/6-项目实战/高级项目/项目-LLM微调实践' },
        { text: '推理优化实践', link: '/6-项目实战/高级项目/项目-推理优化实践' },
        { text: 'AI Coding Agent', link: '/6-项目实战/高级项目/项目-AI-Coding-Agent' },
      ]
    },
  ],
  '/7-技术栈/': [
    { text: '技术栈总览', link: '/7-技术栈/' },
    { text: 'PyTorch核心', link: '/7-技术栈/PyTorch/知识点-PyTorch核心' },
    { text: 'HuggingFace', link: '/7-技术栈/HuggingFace/知识点-Transformers使用' },
    { text: 'LangChain', link: '/7-技术栈/LangChain/知识点-LangChain开发' },
    { text: 'vLLM部署', link: '/7-技术栈/vLLM/知识点-vLLM部署' },
    { text: 'Docker部署AI', link: '/7-技术栈/Docker/知识点-Docker部署AI' },
    { text: 'DeepSpeed训练', link: '/7-技术栈/DeepSpeed/知识点-DeepSpeed训练' },
  ],
  '/8-面试题库/': [
    { text: '面试题库总览', link: '/8-面试题库/' },
    {
      text: '深度学习',
      collapsed: false,
      items: [
        { text: 'Transformer面试题', link: '/8-面试题库/深度学习/Transformer面试题集' },
        { text: '优化器面试题', link: '/8-面试题库/深度学习/优化器面试题集' },
        { text: '正则化面试题', link: '/8-面试题库/深度学习/正则化面试题集' },
      ]
    },
    {
      text: 'LLM',
      collapsed: true,
      items: [
        { text: 'LLM训练面试题', link: '/8-面试题库/LLM/LLM训练面试题集' },
        { text: 'RAG与Agent面试题', link: '/8-面试题库/LLM/RAG与Agent面试题集' },
      ]
    },
    {
      text: '系统设计',
      collapsed: true,
      items: [
        { text: '训练系统设计', link: '/8-面试题库/系统设计/训练系统设计' },
        { text: '推理系统设计', link: '/8-面试题库/系统设计/推理系统设计' },
      ]
    },
    {
      text: '手撕代码',
      collapsed: true,
      items: [
        { text: 'Multi-Head Attention手写', link: '/8-面试题库/手撕代码/Multi-Head-Attention手写' },
        { text: 'BP反向传播手写', link: '/8-面试题库/手撕代码/BP反向传播手写' },
        { text: 'ResNet+BatchNorm手写', link: '/8-面试题库/手撕代码/ResNet-BatchNorm手写' },
      ]
    },
  ],
  '/9-学习资料/': [
    { text: '学习资料总览', link: '/9-学习资料/' },
    { text: '推荐课程', link: '/9-学习资料/课程/推荐课程清单' },
    { text: '推荐书籍', link: '/9-学习资料/书籍/推荐书籍清单' },
    { text: '必读论文', link: '/9-学习资料/论文/必读论文清单' },
    { text: 'GitHub仓库', link: '/9-学习资料/GitHub/优质仓库清单' },
  ],
  '/10-简历与就业/': [
    { text: '简历与就业总览', link: '/10-简历与就业/简历模板' },
    { text: '职位方向分析', link: '/10-简历与就业/职位分析/职位方向分析' },
    { text: '面试经验汇总', link: '/10-简历与就业/面试经验/面试经验汇总' },
  ],
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI深度学习完整学习体系',
  description: '从数学基础到LLM应用，系统化AI学习知识库',
  // 【关键】base 必须设为仓库名，否则 GitHub Pages 上所有链接 404
  // 格式：/<仓库名>/   你的仓库叫 ai-learning-system
  base: '/ai-learning-system/',
  srcDir: '.',
  outDir: '../dist',
  // 【死链接处理】部分原始 README 的交叉引用指向未创建的文件
  // 设为 true 允许构建通过，死链接仅为警告（不影响侧边栏导航）
  ignoreDeadLinks: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/0-学习路线图/' },
      { text: '知识图谱', link: '/知识图谱' },
      { text: '面试题库', link: '/8-面试题库/' },
    ],

    sidebar: {
      '/0-学习路线图/': sidebar['/0-学习路线图/'],
      '/1-数学基础/': sidebar['/1-数学基础/'],
      '/2-机器学习/': sidebar['/2-机器学习/'],
      '/3-深度学习核心/': sidebar['/3-深度学习核心/'],
      '/4-LLM与生成式AI/': sidebar['/4-LLM与生成式AI/'],
      '/5-前沿方向/': sidebar['/5-前沿方向/'],
      '/6-项目实战/': sidebar['/6-项目实战/'],
      '/7-技术栈/': sidebar['/7-技术栈/'],
      '/8-面试题库/': sidebar['/8-面试题库/'],
      '/9-学习资料/': sidebar['/9-学习资料/'],
      '/10-简历与就业/': sidebar['/10-简历与就业/'],
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档' },
          modal: {
            noResultsText: '未找到相关结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    outline: { level: [2, 3], label: '页面导航' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },

    editLink: {
      pattern: 'https://github.com/yourname/ai-learning-system/edit/main/AI-Learning-System/:path',
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' },
    ],
  },

  markdown: {
    math: true,
    lineNumbers: true,
  },

  vite: {
    server: {
      host: true,
      port: 5173,
    },
  },
})
