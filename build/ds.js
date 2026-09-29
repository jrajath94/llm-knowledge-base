/* Research-Engineer Curriculum: shared behavior. Writers INLINE this file at end of <body>.
   Builds: sidebar nav from h2/h3, scroll-spy, nav search filter, localStorage completion
   checkmarks, reading progress bar, chapter prev/next footer, copy buttons, practice-question
   toggles, mobile drawer, keyboard nav. No external dependencies. */
(function(){
"use strict";
var content=document.getElementById('dsContent')||document.querySelector('main')||document.body;
var sidebar=document.getElementById('dsSidebar');
if(!sidebar||!content)return;

/* ---------- collect sections: h2 = chapter groups, h3 = items ---------- */
function slugify(t){return t.toLowerCase().trim().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'')||'s';}
var used={};
function uniqueId(base){var id=base,n=2;while(used[id]||document.getElementById(id)){id=base+'-'+(n++);}used[id]=1;return id;}
var chapters=[]; // {h2, id, title, items:[{id,title,el}]}
var cur=null;
Array.prototype.forEach.call(content.querySelectorAll('h2,h3'),function(h){
  if(!h.id){h.id=uniqueId(slugify(h.textContent));}else{used[h.id]=1;}
  h.setAttribute('tabindex','-1');
  var t=h.textContent.trim();
  if(h.tagName==='H2'){cur={id:h.id,title:t,items:[]};chapters.push(cur);}
  else if(cur){cur.items.push({id:h.id,title:t});}
  else{cur={id:'intro',title:'Introduction',items:[{id:h.id,title:t}]};chapters.push(cur);}
});

/* ---------- build sidebar nav ---------- */
var nav=document.getElementById('dsNav');
var pageKey='rec-cur-'+(document.title||location.pathname).replace(/[^a-z0-9]+/gi,'-');
function isDone(id){try{return localStorage.getItem(pageKey+'#'+id)==='1';}catch(e){return false;}}
function setDone(id,v){try{if(v)localStorage.setItem(pageKey+'#'+id,'1');else localStorage.removeItem(pageKey+'#'+id);}catch(e){}}
function checkItem(id){var c=document.querySelector('.ds-check[data-sec="'+id+'"]');if(c)c.checked=isDone(id);}
function updateCount(){
  var boxes=nav.querySelectorAll('.ds-check'),done=0;
  boxes.forEach(function(b){if(b.checked)done++;});
  var el=document.getElementById('dsCompleteCount');
  if(el)el.textContent=done+' of '+boxes.length+' sections complete';
}
function itemRow(sec){
  var li=document.createElement('li');
  var a=document.createElement('a');a.href='#'+sec.id;
  var cb=document.createElement('input');
  cb.type='checkbox';cb.className='ds-check';cb.checked=isDone(sec.id);
  cb.setAttribute('data-sec',sec.id);cb.setAttribute('aria-label','Mark complete: '+sec.title);
  cb.addEventListener('change',function(ev){ev.stopPropagation();setDone(sec.id,cb.checked);updateCount();});
  cb.addEventListener('click',function(ev){ev.stopPropagation();});
  var sp=document.createElement('span');sp.textContent=sec.title;
  a.appendChild(cb);a.appendChild(sp);
  a.addEventListener('click',function(){document.body.classList.remove('ds-nav-open');});
  li.appendChild(a);return li;
}
chapters.forEach(function(ch,ci){
  var g=document.createElement('div');g.className='ds-nav-group';g.dataset.title=ch.title.toLowerCase();
  var btn=document.createElement('button');
  btn.innerHTML='<span></span><span class="ds-caret">▾</span>';
  btn.firstChild.textContent=ch.title;
  btn.setAttribute('aria-expanded','true');
  var ul=document.createElement('ul');ul.className='ds-nav-items';
  var headLi=itemRow({id:ch.id,title:ch.title});ul.appendChild(headLi);
  ch.items.forEach(function(it){ul.appendChild(itemRow(it));});
  btn.addEventListener('click',function(){
    var open=btn.getAttribute('aria-expanded')==='true';
    btn.setAttribute('aria-expanded',String(!open));
    ul.style.display=open?'none':'';
    btn.querySelector('.ds-caret').textContent=open?'▸':'▾';
  });
  g.appendChild(btn);g.appendChild(ul);nav.appendChild(g);
});
updateCount();

/* ---------- nav search filter ---------- */
var search=document.getElementById('dsNavSearch');
if(search){search.addEventListener('input',function(){
  var q=search.value.trim().toLowerCase(),any=false;
  nav.querySelectorAll('.ds-nav-group').forEach(function(g){
    var show=g.dataset.title.indexOf(q)>-1;
    g.querySelectorAll('.ds-nav-items li').forEach(function(li){
      var hit=li.textContent.toLowerCase().indexOf(q)>-1;
      li.style.display=(!q||hit||show)?'':'none';if(!q||hit)show=true;
    });
    g.style.display=show?'':'none';if(show)any=true;
  });
  var empty=nav.querySelector('.ds-nav-empty');
  if(!any&&!empty){var d=document.createElement('div');d.className='ds-nav-empty';d.textContent='No sections match.';nav.appendChild(d);}
  if(any&&empty)empty.remove();
});}

/* ---------- scroll-spy ---------- */
var links=Array.prototype.slice.call(nav.querySelectorAll('.ds-nav-items a'));
var spy=new IntersectionObserver(function(entries){
  entries.forEach(function(e){
    if(e.isIntersecting){
      links.forEach(function(a){a.classList.toggle('ds-active',a.getAttribute('href')==='#'+e.target.id);});
    }
  });
},{rootMargin:'-20% 0px -70% 0px'});
chapters.forEach(function(ch){
  var el=document.getElementById(ch.id);if(el)spy.observe(el);
  ch.items.forEach(function(it){var el2=document.getElementById(it.id);if(el2)spy.observe(el2);});
});

/* ---------- progress bar ---------- */
var prog=document.getElementById('dsProgress');
function onScroll(){
  var h=document.documentElement;
  var max=h.scrollHeight-h.clientHeight;
  prog.style.width=(max>0?(h.scrollTop/max*100):0)+'%';
}
document.addEventListener('scroll',onScroll,{passive:true});onScroll();

/* ---------- mobile drawer ---------- */
var menuBtn=document.getElementById('dsMenuBtn'),scrim=document.getElementById('dsScrim');
if(menuBtn)menuBtn.addEventListener('click',function(){document.body.classList.toggle('ds-nav-open');});
if(scrim)scrim.addEventListener('click',function(){document.body.classList.remove('ds-nav-open');});

/* ---------- site volume nav: collapsible panel on desktop, drawer on mobile ---------- */
var siteNavToggle=document.getElementById('siteNavToggle'),
    siteNavPanel=document.getElementById('siteNavPanel'),
    siteNavScrim=document.getElementById('siteNavScrim'),
    siteNavClose=document.getElementById('siteNavClose');
function siteNavDrawerMode(){return window.matchMedia&&window.matchMedia('(max-width:40rem)').matches;}
function openSiteNav(){
  if(!siteNavPanel||!siteNavToggle)return;
  siteNavPanel.classList.add('is-open');siteNavToggle.setAttribute('aria-expanded','true');
  if(siteNavDrawerMode()){document.body.classList.add('site-nav-open');if(siteNavScrim)siteNavScrim.classList.add('is-open');}
}
function closeSiteNav(){
  if(!siteNavPanel||!siteNavToggle)return;
  siteNavPanel.classList.remove('is-open');siteNavToggle.setAttribute('aria-expanded','false');
  document.body.classList.remove('site-nav-open');if(siteNavScrim)siteNavScrim.classList.remove('is-open');
}
if(siteNavToggle)siteNavToggle.addEventListener('click',function(){
  if(siteNavPanel.classList.contains('is-open'))closeSiteNav();else openSiteNav();
});
if(siteNavClose)siteNavClose.addEventListener('click',closeSiteNav);
if(siteNavScrim)siteNavScrim.addEventListener('click',closeSiteNav);
if(siteNavPanel)siteNavPanel.querySelectorAll('a').forEach(function(a){a.addEventListener('click',closeSiteNav);});
document.addEventListener('keydown',function(e){
  if(e.key==='Escape'&&siteNavPanel&&siteNavPanel.classList.contains('is-open')){
    closeSiteNav();if(siteNavToggle)siteNavToggle.focus();
  }
});

/* ---------- keyboard nav: [ ] sections, / search ---------- */
document.addEventListener('keydown',function(e){
  if(/INPUT|TEXTAREA/.test(document.activeElement.tagName))return;
  var ids=[];chapters.forEach(function(ch){ids.push(ch.id);ch.items.forEach(function(it){ids.push(it.id);});});
  function jump(dir){
    var y=window.scrollY+10,cur=0;
    ids.forEach(function(id,i){var el=document.getElementById(id);if(el&&el.getBoundingClientRect().top+y<=y+1)cur=i;});
    var n=Math.min(ids.length-1,Math.max(0,cur+dir)),el=document.getElementById(ids[n]);
    if(el){el.scrollIntoView();el.focus({preventScroll:true});}
  }
  if(e.key==='[')jump(-1);else if(e.key===']')jump(1);
  else if(e.key==='/'&&search){e.preventDefault();search.focus();}
});
var hint=document.createElement('div');hint.className='ds-kbd-hint';
hint.textContent='Keyboard: [ and ] move between sections, / focuses section search.';
var foot=document.querySelector('.ds-side-foot');if(foot)foot.appendChild(hint);

/* ---------- chapter prev/next footers (one after each h2 section) ---------- */
if(chapters.length>1){
  var mk=function(ch,cls,label){var a=document.createElement('a');a.href='#'+ch.id;a.className=cls;
    a.innerHTML='<span style="display:block;font-size:.78rem;color:var(--muted)">'+label+'</span>';a.appendChild(document.createTextNode(ch.title));return a;};
  chapters.forEach(function(ch,i){
    var bar=document.createElement('nav');bar.className='ds-chapter-nav';bar.setAttribute('aria-label','Chapter navigation');
    var left=document.createElement('span'),right=document.createElement('span');
    if(chapters[i-1])left.appendChild(mk(chapters[i-1],'ds-prev','↑ Previous'));
    else left.innerHTML='<span style="color:var(--muted);font-size:.85rem">Start of volume</span>';
    if(chapters[i+1])right.appendChild(mk(chapters[i+1],'ds-next','Next ↓'));
    else right.innerHTML='<span style="color:var(--muted);font-size:.85rem">End of volume</span>';
    bar.appendChild(left);bar.appendChild(right);
    /* insert the bar immediately before the next chapter's heading, using that
       heading's own parent so <section> wrappers (or any nesting) cannot break it */
    var nextH=i+1<chapters.length?document.getElementById(chapters[i+1].id):null;
    if(nextH){nextH.parentNode.insertBefore(bar,nextH);}
    else{var lastH=document.getElementById(ch.id);lastH.parentNode.appendChild(bar);}
  });
}

/* ---------- copy buttons on code blocks ---------- */
document.querySelectorAll('pre').forEach(function(pre){
  if(pre.querySelector('.ds-copy-btn'))return;
  var btn=document.createElement('button');btn.className='ds-copy-btn';btn.textContent='Copy';
  btn.setAttribute('aria-label','Copy code to clipboard');
  btn.addEventListener('click',function(){
    var txt=pre.innerText;
    function done(){btn.textContent='Copied';setTimeout(function(){btn.textContent='Copy';},1400);}
    if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(txt).then(done,done);}
    else{var ta=document.createElement('textarea');ta.value=txt;document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(e){}ta.remove();done();}
  });
  pre.appendChild(btn);
});

/* ---------- practice-question widgets ---------- */
document.querySelectorAll('.pq').forEach(function(pq){
  var btn=pq.querySelector('.pq-reveal'),ans=pq.querySelector('.pq-a');
  if(!btn||!ans)return;
  btn.addEventListener('click',function(){
    var open=ans.hasAttribute('hidden');
    if(open){ans.removeAttribute('hidden');btn.setAttribute('aria-expanded','true');btn.textContent='Hide answer';}
    else{ans.setAttribute('hidden','');btn.setAttribute('aria-expanded','false');btn.textContent='Reveal answer';}
  });
});

/* ---------- watch-deeper speed bars: 0.5x to 2x via the YouTube IFrame Player API ----------
   Progressive enhancement: the bar is built only after the API loads. If the API
   script fails or is blocked, no bar appears and the plain youtube-nocookie iframe
   keeps working with YouTube's own controls. Honest cap: the embed player
   hard-caps at 2x; no page-side trick can force 3x, so 2x is the top button. */
(function(){
  var frames=Array.prototype.slice.call(
    document.querySelectorAll('.watch-frame iframe[src*="youtube-nocookie.com/embed/"]'));
  if(!frames.length)return;
  var RATES=[0.5,0.75,1,1.25,1.5,1.75,2];
  function rateLabel(r){return (r===1?'1':String(r))+'x';}
  function buildBar(host,player){
    var bar=document.createElement('div');
    bar.className='watch-speedbar';
    bar.setAttribute('role','group');
    bar.setAttribute('aria-label','Playback speed');
    var lab=document.createElement('span');
    lab.className='watch-speedbar-label';lab.textContent='Speed';
    bar.appendChild(lab);
    var btns=RATES.map(function(r){
      var b=document.createElement('button');
      b.type='button';b.className='watch-speed';b.textContent=rateLabel(r);
      b.setAttribute('aria-pressed',r===1?'true':'false');
      b.setAttribute('aria-label','Play at '+rateLabel(r)+' speed');
      b.addEventListener('click',function(){
        try{player.setPlaybackRate(r);}catch(e){return;}
        btns.forEach(function(o){o.setAttribute('aria-pressed',o===b?'true':'false');});
      });
      bar.appendChild(b);return b;
    });
    host.parentNode.insertBefore(bar,host.nextSibling);
  }
  function init(){
    frames.forEach(function(f,i){
      if(f.dataset.ytWired)return;
      f.dataset.ytWired='1';
      var src=f.getAttribute('src')||'';
      if(src.indexOf('enablejsapi=1')<0){
        f.setAttribute('src',src+(src.indexOf('?')<0?'?':'&')+'enablejsapi=1');
      }
      if(!f.id)f.id='watch-yt-'+(i+1);
      new YT.Player(f.id,{events:{onReady:function(ev){buildBar(f,ev.target);}}});
    });
  }
  if(window.YT&&window.YT.Player){init();}
  else{
    var prev=window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady=function(){
      if(typeof prev==='function'){try{prev();}catch(e){}}
      init();
    };
    var tag=document.createElement('script');
    tag.src='https://www.youtube.com/iframe_api';tag.async=true;
    var first=document.getElementsByTagName('script')[0];
    if(first)first.parentNode.insertBefore(tag,first);
    else document.head.appendChild(tag);
  }
})();
})();
