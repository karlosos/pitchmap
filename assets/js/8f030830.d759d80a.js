(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[618],{3834:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return r},metadata:function(){return p},toc:function(){return l},default:function(){return d}});var i=n(2122),a=n(9756),o=(n(7294),n(3905)),r={sidebar_position:3},p={unversionedId:"development",id:"development",isDocsHomePage:!1,title:"Development \ud83d\udc68\ud83c\udffb\u200d\ud83d\udcbb",description:"Requirements",source:"@site/docs/development.md",sourceDirName:".",slug:"/development",permalink:"/pitchmap/docs/development",editUrl:"https://github.com/karlosos/pitchmap/edit/master/website/docs/development.md",version:"current",sidebarPosition:3,frontMatter:{sidebar_position:3},sidebar:"tutorialSidebar",previous:{title:"Demos \ud83d\udda5\ufe0f",permalink:"/pitchmap/docs/demos"},next:{title:"Research papers \ud83d\udcdd",permalink:"/pitchmap/docs/papers"}},l=[{value:"Requirements",id:"requirements",children:[]},{value:"Environment variable",id:"environment-variable",children:[]},{value:"Running application",id:"running-application",children:[]},{value:"Documentation",id:"documentation",children:[]}],s={toc:l};function d(e){var t=e.components,n=(0,a.Z)(e,["components"]);return(0,o.kt)("wrapper",(0,i.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"requirements"},"Requirements"),(0,o.kt)("p",null,"Create virtual environment with tool of your choice (",(0,o.kt)("inlineCode",{parentName:"p"},"virtualenv"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"pipenv"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"conda"),") and activate it. Install requirements:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"pip install -r requirements.clean.txt\n")),(0,o.kt)("div",{className:"admonition admonition-caution alert alert--warning"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 16 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"}))),"caution")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},(0,o.kt)("inlineCode",{parentName:"p"},"mxnet")," on windows requires ",(0,o.kt)("a",{parentName:"p",href:"https://www.visualstudio.microsoft.com/visual-cpp-build-tools"},"Microsoft C++ Build Tools")))),(0,o.kt)("h2",{id:"environment-variable"},"Environment variable"),(0,o.kt)("p",null,(0,o.kt)("inlineCode",{parentName:"p"},"$SM_FRAMERK")," variable is required for proper working of deep neural netowrk model. Define it with:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'$env:SM_FRAMEWORK="tf.keras"\n')),(0,o.kt)("p",null,"Or with you IDE of choice."),(0,o.kt)("h2",{id:"running-application"},"Running application"),(0,o.kt)("p",null,"In ",(0,o.kt)("inlineCode",{parentName:"p"},"pitchmap/main.py")," change input video path by modificating ",(0,o.kt)("inlineCode",{parentName:"p"},"self.video_name")," field in ",(0,o.kt)("inlineCode",{parentName:"p"},"PitchMap")," class. Footage should be stored in ",(0,o.kt)("inlineCode",{parentName:"p"},"data/")," directory."),(0,o.kt)("p",null,"Run application with"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"python pitchmap/main.py\n")),(0,o.kt)("h2",{id:"documentation"},"Documentation"),(0,o.kt)("p",null,"This website was built with ",(0,o.kt)("a",{parentName:"p",href:"https://docusaurus.io/"},"Docusaurus"),". To start developing run:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"cd website\nnpm start\n")),(0,o.kt)("p",null,"Deploying to GitHub pages:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"cmd /C 'set \"GIT_USER=karlosos\" && npm run deploy'\n")))}d.isMDXComponent=!0}}]);