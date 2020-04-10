(this["webpackJsonpwestern-ai-beginner-showcase"]=this["webpackJsonpwestern-ai-beginner-showcase"]||[]).push([[0],{29:function(e,t,a){},57:function(e,t,a){e.exports=a.p+"static/media/cancer-detection.82b82cca.jpg"},63:function(e,t,a){e.exports=a(97)},68:function(e,t,a){},97:function(e,t,a){"use strict";a.r(t);var n=a(0),r=a.n(n),o=a(7),c=a.n(o),i=(a(68),a(29),a(121)),l=a(123),s=a(125),m=a(127),d=a(126),u=a(128),p=a(19),h=Object(i.a)({root:{maxWidth:345},media:{height:140}});function g(e){var t=h();return r.a.createElement(l.a,{className:t.root},r.a.createElement(p.b,{className:"card-link",to:"/"+e.link},r.a.createElement(s.a,null,r.a.createElement(d.a,{className:t.media,image:e.image,title:"Cancer Detection"}),r.a.createElement(m.a,null,r.a.createElement(u.a,{gutterBottom:!0,variant:"h5",component:"h2"},e.title),r.a.createElement(u.a,{variant:"body2",color:"textSecondary",component:"p"},e.description)))))}var b=a(129),E=a(130),f=a(131),v=a(53),y=a.n(v),w=Object(i.a)((function(e){return{root:{flexGrow:1},menuButton:{marginRight:e.spacing(2)}}}));function k(){var e=w();return r.a.createElement("div",{className:e.root},r.a.createElement(b.a,{position:"static",style:{background:""}},r.a.createElement(E.a,{variant:"dense"},r.a.createElement(f.a,{edge:"start",className:e.menuButton,color:"inherit","aria-label":"menu"},r.a.createElement(p.b,{className:"app-bar-link",to:"/"},r.a.createElement(y.a,null))),r.a.createElement(u.a,{variant:"h6",color:"inherit"},"Western AI Beginner Team Showcase"))))}var x=a(18),N=a(37),B=a.n(N),C=a(54),O=a(27),j=a(134),I=a(132),R=a(133),S=a(100),A=Object(i.a)((function(e){return{modal:{display:"flex",alignItems:"center",justifyContent:"center"},paper:{backgroundColor:e.palette.background.paper,border:"2px solid #000",boxShadow:e.shadows[5],padding:e.spacing(2,4,3)},root:{width:"15%",borderRadius:"40px",marginLeft:"18.6%",marginTop:"1%"}}}));function T(e){var t=A(),a=r.a.useState(!1),n=Object(O.a)(a,2),o=n[0],c=n[1];return r.a.createElement("div",null,r.a.createElement("div",{className:t.root},r.a.createElement(I.a,{variant:"contained",onClick:function(){c(!0)},color:"primary"},e.buttonText)),r.a.createElement(j.a,{"aria-labelledby":"transition-modal-title","aria-describedby":"transition-modal-description",className:t.modal,open:o,onClose:function(){c(!1)},closeAfterTransition:!0,BackdropComponent:R.a,BackdropProps:{timeout:500}},r.a.createElement(S.a,{in:o},r.a.createElement("div",{className:t.paper},r.a.createElement(u.a,{variant:"body2",display:"block",gutterBottom:!0},Object.keys(e.totalPredictions).map((function(t){return r.a.createElement("p",null,t+": "+(100*e.totalPredictions[t]).toFixed(3)+"%")})))))))}var P=a(55),D=a.n(P),W=a(56),J=a.n(W),L=function(){var e="http://lilt.pythonanywhere.com/api/detect-cancer/",t=Object(n.useState)(""),a=Object(O.a)(t,2),o=a[0],c=a[1],i=Object(n.useState)(""),l=Object(O.a)(i,2),s=l[0],m=l[1],d=Object(n.useState)(null),p=Object(O.a)(d,2),h=p[0],g=p[1],b=function(e){return new Promise((function(t,a){var n=new FileReader;n.readAsDataURL(e),n.onload=function(){return t(n.result)},n.onerror=function(e){return a(e)}}))};function E(){return(E=Object(C.a)(B.a.mark((function t(a){var n,r;return B.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return c("PREDICTING..."),m(""),g(null),t.next=5,b(a[a.length-1]).catch((function(e){return Error(e)}));case 5:if(!((n=t.sent)instanceof Error)){t.next=9;break}return console.log("Error: ",n.message),t.abrupt("return");case 9:r=new RegExp("data:(.*)/(.*);base64,"),n=n.replace(r,""),J.a.post(e,{file:n}).then((function(e){c("");var t=Object.keys(e.data.prediction)[0],a=(100*e.data.prediction[t]).toFixed(3);m(a>.5?"The model predicted: "+t+" with "+a+"% probability":"The model did not find a cancer match with strong enough probability."),delete e.data.prediction,g(e.data)})).catch((function(e){c("ERROR ... PLEASE TRY AGAIN"),console.log(e.response)}));case 12:case"end":return t.stop()}}),t)})))).apply(this,arguments)}return r.a.createElement("div",null,r.a.createElement(k,null),r.a.createElement("div",{className:"cancer-detection-text-wrapper"},r.a.createElement(u.a,{className:"cancer-detection-title",variant:"h2",component:"h2",gutterBottom:!0},"Cancer Detector"),r.a.createElement("p",{className:"cancer-detection-paragraph"},"Over the course of the year, the Western AI Beginner Medical Imaging team has worked hard to create a machine learning program that can detect different types of skin cancer. Check it out! Upload a picture of your mole and see if it's cancerous or not."),r.a.createElement(u.a,{className:"cancer-detection-paragraph",variant:"caption",component:"h2",gutterBottom:!0},"Created by Western AI's Beginner Medical Imaging Team: Kevin Zhang, Daniyal Syed, Jinhao (Jason) Wang, Nicholas Chu"),r.a.createElement(u.a,{className:"cancer-detection-paragraph",variant:"overline",display:"block",gutterBottom:!0},"DISCLAIMER: Results are not 100% accurate. Do not use for medical diagnosis. Predictions may take one minute or longer.")),r.a.createElement(D.a,{onChange:function(e){return E.apply(this,arguments)},className:"image-uploader",fileContainerStyle:{boxShadow:"2px 2px 3px 5px rgba(0.05, 0.05, 0.05, 0.05)",padding:"20px 20px"}}),r.a.createElement("div",{className:"cancer-detection-text-wrapper"},r.a.createElement(u.a,{className:"cancer-detection-loading",variant:"overline",display:"block",gutterBottom:!0},o),r.a.createElement(u.a,{className:"cancer-detection-prediction",variant:"button",display:"block",gutterBottom:!0},s),h?r.a.createElement(T,{buttonText:"View Prediction Mix",totalPredictions:h}):null))},M=a(57),F=a.n(M),G=function(){return r.a.createElement(x.c,null,r.a.createElement(x.a,{path:"/cancerdetection",component:L}),r.a.createElement("div",{className:"App"},r.a.createElement(k,null),r.a.createElement("div",{className:"home-card"},r.a.createElement(g,{title:"Cancer Detection",image:F.a,link:"cancerdetection",description:"Over the course of the year, the Western AI beginner medical imaging team has worked hard to create a machine learning program that can detect different types of skin cancer. Check it out! Upload a picture of your mole and see if it's cancerous or not"}))))};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));c.a.render(r.a.createElement(p.a,null,r.a.createElement(G,null)),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()}))}},[[63,1,2]]]);
//# sourceMappingURL=main.92a6eb3b.chunk.js.map