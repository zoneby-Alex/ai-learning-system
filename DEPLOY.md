# AI-Learning-System 部署指南

## 项目概述

这是一个 AI/深度学习完整知识体系的 VitePress 文档站，包含 137 个 Markdown 文件、49 个知识点卡片、10 个项目实战和 100+ 面试题。

## 部署架构

```
本地编写(.md) → git push → GitHub Actions 自动构建 → GitHub Pages 上线
                                                     ↓
                                    https://zoneby-Alex.github.io/ai-learning-system/
```

## 前置要求

1. **GitHub 账号**：已确认用户名为 `zoneby-Alex`
2. **Git 已安装**：本机已有 Git
3. **Node.js 已安装**：用于本地预览（非必须）

---

## 部署步骤（5 步完成）

### 第 1 步：在 GitHub 上创建仓库

1. 打开 https://github.com/new
2. 填写仓库信息：
   - **Repository name**：`ai-learning-system`（必须和文件夹同名）
   - **Description**：`AI深度学习完整学习体系 - VitePress文档站`
   - **Public**（选 Public，GitHub Pages 免费；Private 需付费）
   - **不要勾选** "Add a README file"（我们已有）
   - **不要勾选** ".gitignore"（我们已有）
   - **不要勾选** "Choose a license"
3. 点击 **Create repository**

### 第 2 步：初始化本地仓库并推送

在终端中进入 `ai-learning-system` 文件夹，依次执行：

```bash
# 1. 进入项目目录
cd D:/project/claude/job/ai-learning-system

# 2. 初始化 Git 仓库
git init

# 3. 将所有文件添加到暂存区
git add -A

# 4. 创建第一个提交
git commit -m "feat: AI深度学习完整学习体系 v2.0 - VitePress文档站"

# 5. 将本地仓库关联到 GitHub 远程仓库
#    注意：将 YOUR_USERNAME 替换为 zoneby-Alex
git remote add origin https://github.com/zoneby-Alex/ai-learning-system.git

# 6. 设置默认分支名为 main
git branch -M main

# 7. 推送到 GitHub
git push -u origin main
```

> **注意**：如果使用 SSH 方式连接 GitHub，第 5 步改为：
> `git remote add origin git@github.com:zoneby-Alex/ai-learning-system.git`

### 第 3 步：启用 GitHub Pages

1. 打开你的仓库页面：`https://github.com/zoneby-Alex/ai-learning-system`
2. 点击顶部导航栏的 **Settings**
3. 左侧菜单点击 **Pages**（在 "Code and automation" 分组下）
4. 在 "Build and deployment" 区域：
   - **Source**：选择 `GitHub Actions`（不是 "Deploy from a branch"）
   - 这样会让我们的 `.github/workflows/deploy.yml` 工作流接管部署
5. 无需其他操作，GitHub 会自动识别工作流

### 第 4 步：触发首次部署

推送代码后，GitHub Actions 会自动开始构建：

1. 在仓库页面点击 **Actions** 标签
2. 你会看到一个正在运行的 workflow：`Deploy to GitHub Pages`
3. 点击进去查看实时构建日志：
   - `拉取代码` → 从 GitHub 下载文件
   - `安装 Node.js` → 安装 Node 20
   - `安装依赖` → npm ci 安装 vitepress
   - `构建 VitePress 站点` → 生成静态 HTML
   - `部署到 GitHub Pages` → 上传到 Pages 服务器
4. 等待约 **1-2 分钟**，全部步骤显示绿色 ✅ 即部署成功
5. 部署完成后，站点地址为：
   ```
   https://zoneby-Alex.github.io/ai-learning-system/
   ```

### 第 5 步：验证部署

浏览器打开 `https://zoneby-Alex.github.io/ai-learning-system/`，应看到：

- 首页 Hero 布局（标题 + 搜索框 + 快速入口）
- 左侧可展开的侧边栏（10 个模块全部可访问）
- 顶部搜索框可全文搜索
- 所有页面正常渲染、数学公式正常显示

---

## 后续更新流程

每次修改内容后，只需：

```bash
# 1. 编辑 Markdown 文件（或新增文件）

# 2. 查看变更
git status

# 3. 添加变更
git add -A

# 4. 提交
git commit -m "update: 更新XXX知识点"

# 5. 推送到 GitHub（自动触发部署）
git push
```

推送后 GitHub Actions 会自动重新构建部署，无需手动操作。

---

## 本地预览

如果想在推送前先在本地看效果：

```bash
cd D:/project/claude/job/ai-learning-system

# 首次需要安装依赖
npm install

# 启动开发服务器（支持热更新）
npm run docs:dev
# 浏览器打开 http://localhost:5173

# 构建生产版本（检查是否有错误）
npm run docs:build

# 预览构建结果
npm run docs:preview
```

---

## 目录结构说明

```
ai-learning-system/
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions 自动部署配置
├── AI-Learning-System/             # VitePress 源文件目录
│   ├── .vitepress/
│   │   └── config.mts              # VitePress 配置（侧边栏/搜索/导航）
│   ├── index.md                    # 首页（Hero 布局）
│   ├── 0-学习路线图/               # 3/6/12个月学习计划
│   ├── 1-数学基础/                 # 14个知识点卡片
│   ├── 2-机器学习/                 # 9个知识点卡片
│   ├── 3-深度学习核心/             # 13个知识点卡片
│   ├── 4-LLM与生成式AI/            # 8个知识点卡片
│   ├── 5-前沿方向/                 # 8个前沿方向分析
│   ├── 6-项目实战/                 # 10个项目案例
│   ├── 7-技术栈/                   # 6个技术栈指南
│   ├── 8-面试题库/                 # 9个面试题集
│   ├── 9-学习资料/                 # 4个资源清单
│   └── 10-简历与就业/              # 简历模板+职位分析+面试经验
├── .gitignore                      # Git 忽略规则
├── package.json                    # 项目依赖和脚本
├── package-lock.json               # 依赖版本锁定
├── DEPLOY.md                       # 本文件：部署指南
└── README.md                       # 仓库说明（可选）
```

---

## 常见问题

### Q: 部署后页面 404？
**A**: 检查 Settings → Pages 中是否正确选择了 GitHub Actions 作为 Source；确认 workflow 文件路径为 `.github/workflows/deploy.yml`。

### Q: 构建失败？
**A**: 在 Actions 页面查看构建日志，常见原因：
- `npm ci` 失败 → 删除 `package-lock.json` 重新 `npm install`
- build 失败 → 本地 `npm run docs:build` 测试，查看错误信息

### Q: 想绑定自己的域名？
**A**: 
1. 在 Settings → Pages → Custom domain 填入你的域名
2. 在你的 DNS 服务商添加 CNAME 记录指向 `zoneby-Alex.github.io`
3. 勾选 "Enforce HTTPS"

### Q: 如何让站点支持 PWA（离线访问）？
**A**: 使用 `vite-plugin-pwa` 插件即可，需要时可添加配置。
