#!/usr/bin/env python3
"""RL-PPO as a standalone Axelrod strategy.  The actor-critic checkpoint is exported to the embedded deployable MLP form; only sample mode is supported.
Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
Use the module-level ``GAME`` with axelrod Match/Tournament.
"""
from __future__ import annotations

import base64
import json
import math
import random
import zlib
from operator import mul
from typing import Sequence

from axelrod import Action, Game, Player

# Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
GAME = Game(r=6, s=0, t=10, p=1)

# Internal action convention used by the original pri_dile policies.
C = 0
D = 1

# Axelrod action objects.
A_C = Action.C
A_D = Action.D


def _histories(player, opponent):
    mine = [D if move == A_D else C for move in player.history]
    theirs = [D if move == A_D else C for move in opponent.history]
    if len(mine) != len(theirs):
        raise RuntimeError("history lengths differ")
    return mine, theirs


def _require_sample(mode, name):
    if mode != "sample":
        raise ValueError(f"{name} supports only 'sample' mode")

_EMBEDDED = (
    "c-"
    "maO?XI;=a+G;4{kv`5^<BlOAcR_s(O6;(GKLZ2<R`Ak%=I*5<EG#J?!DIiQI(Y$84>xv|MH*z>%aa#fBUz;{JX#W^MCul{^4K#>3"
    "{s|-~P)#{I~!5U;g<&{jY!cpY1Py{mXy;r@#H%|M=H``j7wfpZ@i~|MP$Sm;dt@^I!h=_|<>;kM+-"
    "g`FD5y^)G+>PyhDM|MPGE7Jv0`fBTpJTfg&P{`SB0L4Wyo|Nejf<^R=x{`>DATf5^s=j=0I8}r+L{Re*d_10e7eV=>U-"
    "GBZ5@tf^y-#5Q=-<^HOZ+`84eaw5toa4^+2EV%2*IMuM``VuSoBV9F^Uc%89rL|5*YYpkz1n%>%-"
    "iNS#`0%weJJ0u$K0p49Y6hU(|_IlE?w`a-~IH*{q5WP-"
    "lz5U_}F*X=(CS;&Nt5Iulnv;S0B;u*M{ztpXr;&)phA#=t@`fC!W{mcyIOj`f0--xVrwmo|L}#&98NT;(bSt<mra=nO~cJa^KR2-"
    "}UZ!Pao`G)KfgWAN|%?Q+I#;IPcv0Kc2wTe|i4yp)Wc8d+pw*EAvHe-R|7?_(R(D&$|Bmrw7^f)Xsj-FQ^^8Dm~4%+HC9ge3+|^b"
    ";sUZ_8HwjTuTr3Uj2b?QI{F-a`67PUXk9v{%*!kmmaO&-"
    "hSU5>#Y&D+0N28jobC2c8gE_=IGyPYptnAx>k?1)K*_ZU%21ex<+3=Z$=BF-@f^zSW0U?y<WZJvCiuI#-"
    "Gojr8DQ&tJ|adhkBxVF}>?*wk|%t=$o_a&flkXsD<2P)wa=}{#?&}@2;=SD_xC6rx!I3A3Wnur=Gvwv=)=DDSvh7h3u(?u=VQll^"
    "k8vW;JlH&r77~F=_$oS?dkzDtzsu>ov5aPixvSzRUK`<r^+n`QD5*xAo4o;CS_)R(y*M(;rJ)3x0PkMX$s2-"
    "K(EWUGp2CH`~!iX{j&0B;8+&x2_x320B`n?S4BhM@#tA-"
    ";A?&TiS~=caYw1*L!=~v3m0>7lXb=?_68%Xwj}a^A$B+<XgIjsRjOdC2KYIrES4N&=b<T{%d~h$Ntq8ysy283{Q(d7pV<8^|2XPj"
    "&5)_Z4Q0pix=!LqrI(Hp)hhseBvHj$5StKXhZ71$9^693T<<(DE2Gw;?sJ0eE)6wvJuZ%-"
    ">bbsWN2%@88>vBu0(rA0c+_Qwg`0WzoVDd2|lqnd0SJ<O;2ggmV4{zA?Q8mHED(WS?H<hYqje1!ABP!->m1b4)@4q-"
    "?;3lAf`8`%hy+TU-!p?(BsyPXwmiKr=!bKm>_m4E-5VKb?XZjYgOz2DE!6q(c{)H@tCx}VyP;4XuWk-gC4G~HX?<B?baTftkCH%("
    "(cj^w4U{t#>gwz<6pWI{jOH%jIYsJ)B_rd6o-KEaurP-0JDDt=%;<reeGMpa%m6vw%3<WO{+l9Q|p9y-"
    "LAh}J3%YPn?<k8f35$b&Ai9c*`zbb8$b_IYeO50(?Y)x+w6_5@TB`6XX(_6A1a*bO(;NXFDUZ-hrj-NU#B9tc9wSjAtJ@f(zDcKY"
    "&;6qL0l0Bg^mn8eO+5Vj;+n4>()M1c#ZJZb=?XE+D?j@`j-"
    "*#wN13d?$XkHCs&Z7g5F@$Z|U^Q4b`6_pf=U)JT|=$KQ^r|t^TgKuD6$)cj@nRi;GvQkMthnklm~+-L0-a9>>Qyp-"
    "8C^rz<+~#ubpYbhmC>C(9Q>TuX{y&@OCS+j4$+Ch8zvI#55K{muL7vO{#R=w#KW#xqkC(G5;5w#&U`e7IU3Q?HnXH8QShoj=_q7k"
    "l+@<0=(kpO&09|3~DQ5j6C%N3Xa$?9L3qdgY3;TBdrupQ4WUg#xPrQP)yq56!qEUYWwB4%=TaP-IXhF!dQakhR&m|5{I1t4G(Z57"
    "#%v>pjF_ot;m=ubA!Cu6x$8HaS=HxhK9xr>^oJ9XxvHPv5`uO#JTRkYc07p`fM0edq}*<N10<VIGH`i7uX1pHEJIrekTa0k_U3Ry"
    "#)#DH%7TFf+X@j~<o&P!V7G#m<9E|H~KXIqK_U_h^CYx;ICnVp*R1m&iy|*B;P=iIt^ei}Ov%0qZi8pv`OKAGNSYtc=O_K8qm4*2"
    "-1gNg1wwU0?lH25LR?$04Jz$M5EuyGg=y-"
    "s^d2v*jDp=jq_rf9gDKzI0ug*2&N^APai2@!KUfviZovzC7|3EcI>%Z?S1@?l_3`2$tRe@AHctrL$sbi)kBhoSZx(6@rzj=mh6rm"
    ">D<7jFc$oX7t@!IdSbeIJ=V?r4!nA86K4d=-"
    "KIz(FW9V%#2vc<lc%~3fz0Gh~nB6Pp9kVJa6oI@AIh}ym~HLUGv5XKRB8dQX&X^k<RLq^xcZ+-"
    "29BJaPe;Saf>W9;|5vzeLBIFf9ewAat=8F2NFlL&bl}No=-"
    "Gl>D5DteZnv3C2hiKeBkEw==foEDUpj0WHH`NvNwsaweuGHTsu#D<wS_kqG!8mD=WbraotA?hWu11#UfjZdtpJc8-"
    "1PeY84VVP_(_c!L{<(YufW#!1|n)2kBbWE&3NatXb`Gf1k59JwQF|xEP)1ORqquzt)SQa_%OD0WDU&Cf)4LV@GfEDo}8k`310!$t"
    "asHMCWKe2t6kqujIFr4VwYID^zk8>OXj`mELhuE4cZ0Z@w4>be(v55yY`L*tN8DC8{4N50qA|xFY?EzCZ;BrOk{0rd)<Hz%S;zI^"
    "I~koI!e}iYkgDy`ChA)ZzTmd+V`pwT}qO`kG0!Y7y)7$aM3{YxKX`2HsAc=t1Du;j>>>3KSp87zwJ2{5Er>r}I}KjNC`{QfyHL`p"
    "fH8aWY=M9;NR}RSq#rcajlMuS-"
    "X!zC!O^pA=ORyJiq`RK#TQpd3);kXEm<!;iBrA17(H9;e=!7ILgWjyk=EtHZ)+l>dAZGqmye?iOKH>t5Lim#R&qoHt*a!Vk3}N33"
    "#-N;kFh6k1g`xzNZHPA~oNG?Zd#B}X9A;~6iiCmvY646OtTE!|^mJgu0=g6R<o=7~L_)Jy-"
    "0#Pr7N=`0uC4^^mEU0nA^sjhvhD|mUp@K_Ir$U79<5tH@)1}QhC5&3EC9kP4<`MWCVzgat~QFAKAB+LBr^{C)f9MQv2?V?cXZ`a>"
    "wxoLw@V1AitC_0a)K*Q-lf{~>qIiSvV{Y2MUuUv22H7kX^qNwj&>s&Ebi$F)K7JQFout@n;vhb+$GOyAr1sEkh`p>)~-"
    "Puq8Y$RUV=Pg!|&K5mko#-m+?#il`9H;4Z^or)XMlcG!RHdz>M~dYP@TmZw1c-"
    "=rT4`I6dlRozx3t)!oIBdPiZ7SMl;}dV`N+Q~xoU2E^3=<l)za1n$0pN769qp%&?xa6X&E>BvCJd;RVJ<HqRs4_Hxssx!*Oy5%nF"
    "_g7X0#d-"
    "qv#IY1Jw_K71y%RF=0oQy%L<8HsY`EdB`i>`&tniX>4==t$G^A;>(E`iOUY!sT~K2&2@ZF41GqqtpH(WabmsB3pdTpel?C8O%Y8P"
    "+LmbT<`JN6r9Myqbkn5sFVQ^L$|MY+2e=$O9Hq8vog^2q5{)}uyu51>WtHa^EswdcaS?%)vHS02{Y;cX+azw!Od-"
    "=6MJzA4d2&x7N_vpKrR$_fM=qFQhq>S(mGPwl^3M%Qv}p%R)t7r-A4YDV{L$A=-"
    "b<<JWyrUcG%Ghd|pLYzoacWNEHf|;Dpk>>4?x1QWg^*sbWk;EbCH9iq-<xsRZqE(k=@6m-"
    "vTnNZ}g1g1d^8SeKz?slQe%P^NqGBSCZ&@7{IgRcShTyiyewnS$<LtK!zC)nnHo5%3yt*L2~{MI<<<*PzNmYjE?>d)6nDQ%vi>Q-"
    "$i1^6=1ctGm?G(+BsK>rfZjT-<9DD5H%cPDhe<@}-m-"
    "EZ(eX!3z&r#p2Rt?5j(yOWpQ}qy9yuO{5y%h%+^dQqlz_t9o2M)_#xvxYE4ymr$_-"
    "8u_wzmtN>B)R4ji=#W|};x&6vVM1x}q@a(tu1BlvUtxf%bVkCg6?UBLa067pfsIn|5Va|4GjAifD4J+fk|pGBR~fab*(dl<&l-"
    "hVjR!p<%gC$LNuhEg(2+lg1u+*4Ln%W<Q3c+2*s2P_+KQQ#YZd9N(D|zR**inuyu8-"
    "|gbkWG+NyBlnA7n!*^G`{Ix9ILj*epGr!Q9=g_Z7m5(c!jB3A1;=s4RT1Ob&&&neV`;}J4rpI+XH0`}qP&T4w`Bq;T$w_Yux4xl^"
    "%PbXP*oUhjpTCatry}g8#z*B;2&FD!cIHoYC%h9Rbs3&LSUV7RJ_bT%^%3j>`U?=EAQQ}5gmhckMpA6*y*{!PNb!-"
    "v`RF9LyCQ3?`2Kb=hMMV@#jT}!AQ0D;US(U?w`Wk#)u}7gHao!8K)C{g`C$0#PlYplWr+ob~<N^;Vc4-~+tu0}6W3-"
    "$jfH%#tWApiw!$-"
    "F?GeOmI*0NuuiYhY#4SvKuK<on&S;vhwx$f?f94q+M>36}BhjWaThxTyip#fHCY2?8>368)Y3f!GaD&M6Z4o9LsUP)BMIDId;?IO"
    "n~H>?&0a5q=V8Qp%Tl~MulRPDkUr-"
    "T0G0n+63B%_I+>X~iwg#ozp#op&&)2`B8aWtIR=2~l<o1l2T(h%_fMv4A_a9f_LMwo{2tyOg`8;xM#611W_NgK7967fx6Bq3eB-Q"
    "xmluyV6gvwY@UiW86@e<leo4{vZGxso4oeDLtKuU~>NUKxtS3Nf4;d0tWI>(g|Ek~U@yqno7e;*N+2H&Ry)+U<-"
    "P{JCXkqwyNP&!B3CBLE>?9)}jy^8J2*GVsgzoKvqve|2<HM?ESeLVT{L28N)dfHE`C8LDONqDh2a!1=zYOT1NB@R!}Hc(te}g^Qk"
    "wUj50!ih4u$xSUsHBqyZ;=hfP&LzMoQE?*Ci@R`uZCPL{_r`E!aNAJp(b`3@O%BppIMp0CoDpKjLj8Q9hZH}6Rrbs@N%xh&X7lc5"
    "tuIWA=C828rpw3Jl^i+ViKROF^%Q6K8Vf&h6d3Lx4u>c5wG$of8#qu3Sx>S&!nU~o)c@#n?EBW4;4K1!%83Eva|CQl70Q9z4UhUg"
    "iR_W?!1`&5hfqau{ofP0Bn~G>oBs2;{4<z<W2ffz{S=7wCGLl6^12WQi&sNMrPeEIiKAH7nLB}Ho;->J`&1?>&Jk%9;0k!mg`^X-"
    "fqOr>jx9GA+r2r$N+Z>lW0E4bSR$OSrwCQ$`*OsCc;di(fkch~n9nt`(bfpO{DVI_`*Bn%o5qMvc<to3UWY>q}FFwu^y^Blx1;ii"
    "{f}tByq|kG{)oayG*DK%z)h`p~3kaebRG`t1U!1mzP;+zSQfOqU2*&UM3OMJ?c&vS+bes%!x+i4NlD((*4Wz8|Hd{P;IJ8Lt#(IP"
    "<EnU5bKxuTyco{Ts>yOk}D^MXd7n}|fMRs<cq8&+|R}8Rw*AC+RZiyi*)oGnnujecGIi#~rv4gJkF1X-"
    "2MNzQUcIhYaU)8ZZj%$!l@6!(h3{o$S?f|{tI5G}k&}8{5KdwViZ$=MRany-cwkyac568bApO$MJ88k167<$IEjx$VjwxX_Hp-_o"
    "abjP-"
    "0gjTZhj1Pcd$BCm<Nqe+|K7O=@0xZ^_O@M(0*5N3j{HX!@k$RBv#dU4~W#xb>=Jf1O{gfkDA4pH;MZ>VUQ%V_k2sH6MUcO{kJG+2"
    ">9;ADXk0l$+esTjZ)>r7p&MO3&dK8F3ZzsVM<qX@cBE3g~T{WHuwI^`Cs(=g<q`Zd$Q0rS!D8T{nVh(>YqTRE#;UMN(UJsx}TFSe"
    "H9Y91>trQ=rj6>Vy8EuT5R{LI&RnJgI6X{RxaV1^)u1kc?>XqD4%c84#Bn`;RF+Q%lIrJD#6cUDgX*%m(XQo^L<_&C4OP(%4CHR9"
    "jqk^WYz+DY%y4xT>Y&1o<!0?<E`jpnAy9vxS_y1O)Cu`@y0Ma*%V(8Z6=+pB9e7tnk^|XDNoG6ce-zN3TXwNq8dI!8st@$iUH*JI"
    "c8Ma`%Wn>3GRHdr(;rq2}E`V^XEM0!qx{)1#_38+?w=x{M@-"
    ")|IMfwFP_<{Sg;I!T&=~3><Y4*Tb^?LzcYL&520SdR?>PfteOiiy?!B7jSSEEKtVM9Q8Jnw9D(i^AxT)=CUG!TY4J1?w+Y+&ji+!"
    "9!v=e=kS#DD!V<=v{JTBppM?q3ERs$7env=ck>i)2G=ay2@^EQeNUI+sWrn=a|j1BnQ(!&JLMNkx`~L=d;P8t`A7U9NiR0_t?2D_"
    "<0cO#un>12CZ~t4qFf0LTKb(1TDNIygWTql2FCo;RlwT1xDjS-)x;I-"
    "UAv?ZI8BoR$$yZk@bziIaIi*HWLU#KYku&U_{8R3ISMl%#$U6O@ofxfZ?hy38Lhl{lr;qLUe^fDUDvYd~|<UJ;!d`*9x_Jvvn5QS"
    "uD+zmuAFMV-h3(n8hml7mF%2_6Ay0ZfK7G@CSwzTBn6<&1M$bf_eh33nE10xe3qRVhq5I)`jKwuwRo-"
    "58zcgmojk%ErJ^b)d1&R?8*^9RfOW7liFb^kzLh&Ip?Hl`T*Of3$DN`cFJbMKwh$&UPBVGb1{ADQ5trj~(ePH5<Wi(?6U9!AO-LB"
    "azeTcSl4G@L$Dsh$U2OapNQ-"
    "K%Zm~i9V3XOwJkYY!#8!%GFjHUNp3UUc!YSHVU#%o%V@yg3jfnlX(e)u^Y9#X+bGM9^XMSFtnTu`qJ;MJ)zWvZb@eyYKO)a)k7kS"
    "eS|@B$f&-"
    "6$vH^rCj<_lQ@WE}(^#WN7Au1kZva?7GR5r95(0Fk3J)YjQ7E}}=KM@$Wwt|m^hQ`Vi&Qf~t7a;L?w*@H+0xWCclZ}<Dq?6Jhu~L"
    "@O+^PUzRBc`!@~oS%Sr<zWI;5c+1CplquH%htF@^(ysA!O@9S;ohy(>3MOCAfa`@(!IVMRNt#xm{o*ab%V}it{(|ydU1hx`c@9>2"
    "cYGgPP=EkFlD{O8<0As}fw<jp0Jx@!5HJ8m(#mnO$4+_pH8g!aN#~Xk`Csw4BN--"
    "&0!Iq8aN3&+Z%Ti$1GrPH?kCQgl*c&i0X*X<bHXj%@?cY3nSQXpnCpBvHOq4E?L(m|kKn%VJ9TQ3qEg2BmJYvC?fiFQqvLn!qyy@"
    "0~5wzf=PfDk=uP*?&NUmw1D|dm;_vlcrU?DA5%H&ySfkrj=2-MM2Q(79STXTK+aVL?P8*RH}#rkp`L8I7T)Za>N=>rytLjlZpp-"
    "cn@Smrv5$f_l!gJmVXk47UHjdmsJeqj>S<1Al=9fEKqJJhdKj-;A_ax~6X?U&1%`Evb(89h4pu7&zv-#<PecA9t!xH=a8=^wr-iw"
    "7_=@0NUSW!zxl=rGqCp1Y#sf)~P>plmLBE4q<K!9~lFXq-"
    "=1$;$Ei%l73nX#>^>eSvJ;LdASs<D#DHv2q`&HqCOX#sa2P@C**+MP-"
    "^e=xeT=7e}^YUOuvA@`hIQal#&tf))<wxxSH@8?~9AGbzTJ5IEyZpdmBqxpeZZyz2?5c3r9idRK;}MbOC)iLDV>=^~It&wO8AKb<"
    "DIBGZAY6pSj;Ou~mw7+q`r-"
    "UHVlr+sb}`Jr7<dt+pKMwU*}Jt!yc<%lyCrNKH>z}J^NQmmw~LU82DwM&D8HWfkofdr%!VrH$TPy+8nwFpi99j6YJK3JHRDm|g%n"
    "rbT@dYK1$Kp~q>?{DWS<H>?1)4e3e=P)vS8;h9c3GJWE@U^yCk^pK{4GG9lWl`Q!7_}LM86fBf!wxb-"
    "76?uWJ4MFjXe+1{n1eP?=hTe`GR2o-"
    "q~0<Fq{z&s%Uw{(6+E{O)&)F6S~g^tx&IYTs6W6_^(r%#LiEvLPTO25eiox9M?r2f^DuVTp@9$I)+kp*UE^e&J|GK0$I~0`^j|w!"
    "VM5uF8_*;tEe*PX?sS;E3`1^2e1pH|cB0G1_y#Ci@B%w+q==1lRffC5G}3$V(O^EhW?qyMam;m6RJzL~BH3($3#MxW^Z|UU#>{p0"
    "{am633PV;gNgp3PgW@?TKF}n9rBFqIf;7xrFfk74L|lxg6US+r=2=WadCY2K7X@jEJ|sDUHpGm0!g<5_q*uR*gwb&{v29XexbBY5"
    "6`+V>3Ef3f-hg$W7l5J;sFA_wYChedffB@Dr3P$y6)I*$#*P8%oX~8^D!)9UT=G+1b;vJ?pVBs;EK#k*Z}j*Yr6rJ)-"
    "FLjzpkt#BKpQf$FNlw{@W@j1D4~c1WN@MODZT9tHb<OgZY^KH+DFeanNE`=T8GC1MQWLV&^w351oFCiLTStSty!d64avGxd98I#L"
    "w8nIp*2SjVwq=r6WGw|>4Y8w9Lb3qfVtbN9FaoezMSH;vS=SC`)|8P`beTjb@f5foy98E+X;$H67|n)po$1kr^ql=Hhh3gdC#1Si"
    "f|XI*39Tg!}znlq6d-d(inzn!GToepDq||l&b*>i-"
    "TjB6a|P6y{J%e&i4(opC;o;3fbUbetw##3I~us0DX|?JAcKjqW5z`MHagfx-"
    "r;I{BmAux>@76B|6imN(k>l4AwE_MrbfsxWEPL06oT?K^5Jc(ak)M72*jk7zH8nMJ?y-$ita#-"
    "I%|MncL!I<RR+pVe_0gmkH`n<{>GJFM+adBJ?q;*jgm$WEHUU*xj^n1H4xB=M00&P8Ju7aqvGn=Cxf;|7{skR<I&i5Jk!~-"
    "DS|NAwSE)h%a)t`?)WW0ZF9`aL-"
    "1r@Z()IA6k&{w54`6a0kJgBuHx^mmZNrIf1sSB0=Wo*I*4z6ChSUBJh9yQ15`eIa6Hv%u4y`k&@FUw_&d4bdL)<xsfit1#ku!XJ6"
    "FO8~ADywUxMx%)+MS8z^v8=Z-"
    "8~sWmvLjywO=qSOfm^85gDb!%C3gZzUhLZMsvtKbR1Ebtg~#3wEEK_aA42!yutR6=n;4;X^hOIEbj9d+3w-m(iaa|Suqi=-"
    "(2f<!SvskLghnY=IQG}mf|qiy$)f<#yY{328;hybHS<+>m#l7}18Pf+eqr4H(TJ=ScVDIRHS6Z8q_0Y{RNzUQ4gIN;;T2V@gYSXv"
    "j=asA*Z=`js!ZbD~7Ge|F<{}|Q%CT7y4+wRIIS%a5=l;lJAH0#g-"
    "?1pRM3_3Hi!~YPEt*`hBG=)b`N_1Kg8$gsS%~0aCVb(?AszB~X{Pb^D@N*Ogyi-XIqg70hgmJ6SI4DeVanRJ%f5L*+LR#4z<qZHm"
    "0g**|F}4ZOBx|E+Y!FR1i|^6Wet~0=V1wi8kv6c6)SxYkS~}ZjBnO`txrtfCu=VA7=>|6kyg#smpeuc51kw&x9MnyuRN&pf$pHw~"
    "FLP{wbg-UH&Q6?$Q+fI!@u5*fHOk1wD+S6eUKVXo#9Z(X0<0!%p-"
    "17B?~>{$pd3DHKqOARS%b3+JIBnCk@OAv7XW|Y4?7b$;2|(0)4kjI!I3SBZ9VRcdg!p02@;A=5JCf$<6$a{>z=fclP&cbu5tv|4j"
    "9Q|bu|Gw=@M9esv00tMKm3#dR5iU`;1UM)OvhC0KxhREDBi2*|{hEQgRE?(dpev*!57jK~mCFj=T#{jYcmU@cW9?{aJ2xSe@Cv?I"
    "c$psQe_UN*IlR!#m6hE28t>fm}eK{Ibf~v@F?AkDs2RHqkbwIilzG#c}w7FVS|ND3an~YC9^7n>7KZ+N!_Jp@zf+F-"
    ";^d+?LK09oWs4Y$!n%vP(CK2z*K1r3VVdp0uWz4T?9LH@;{v*82rh0X*LHutrLmz?D0EORBGAHW*k%Dd9zUGs1CilbjLc;S~JxtS"
    "?!H`*zk5ES5vstwR<PTa+b>)WnbnqDaRM8hCAA*GVHOmc4U~=wG19OF$EnMUUPmpAv^B(NckUUAM}=VqP5x4@z}mvjw9ZO`z_W$Y"
    "4ETrpkIUs2%78r5Ho+n%2yNA^3$5iZVszFO|SXFjWu@^<Q^=Lz~w>RO;-HgdkUi#IqC*1{Dd%%xs71-"
    "vRci!cEFCi~!Uk5X|NX$<qo7_JsiZ^%71js2|!otiX|LdXPzSbXd$4DZUb)<s!%cP?r3(y3LC2fWMi-DA#a<4N_oKtI*X*5h=`+@"
    "B}&l+D}T{ARf@o$ddx%7MZCksAYz5&_l_Tq2rQuWx5QbnRFE{EOfMD@c=I_V5@8;pg``k(|ALt5dgOKlUY_lp-O_h>O)vFKF>!Gt"
    "-Z?OI=k&1zXzLv1+0^5(~tORdsT>BV)|E+G1$W@yDdM8BO4@Pz~oY{kZh6W)0K|jq-"
    "u!pi{t_U(t+yxX0~oXhsY#}lr2u&McK5Wbx?4HtcohDTMA0s`6UFxMpdvx5lJDPh{JaAK)YQlKh-A|>vqFq{U%VnVB^f;L-"
    "v>az@1k{?-shw*S8XjL^rc6#6n+})O>x1@@O(GQ?-"
    "JYiqiR_yeTmLGmYr6pXgD=j;C61%fWJ~H51D}wA+<F@wQ(sAwAedg{oCV@;XCiW=d-GLb6ACO^Ln8!ob)IjHg=XQR+)L_fiZuS-"
    ">Ij9HW%AW~KCjG&yxC)AGDW1Bb?OiKwn+pC~M#)ny9PNL3((EaQ*)sT?Q3$L@==5%byW0w7Nvil;yzkvSNiQ3PDYTsQPYE0twKiz"
    "MSy4+JRCv@ABG<E1U9)tqsRTz~RBbg+zo!{DwXaadH75d%4zc@DJJ5oQPWNf_acfYW{fd@UY2AE9y|VdGyDo+0%5{sFQh0U5M%k+"
    "#r4%^Z*B1NjxaoiEO<#ton~A(qZ+4ug3?`J_4=W>J|26(9|r#Ow<4z&YR^T!T~3OphC&gu_qQnPNKfLhaDyR4q$d9&D%mj$VHTbO"
    "Jg@Yha@3rPu9Bm6VTM()|3Hunp|O4av6w<^SFccq9mq#gdsT=sHL&fR5v9K8y7rA%jGe@U3y43{W@cWqGzj^VVkl%=b*L2<T_0NP"
    "*<9#W)hldlsA{Urgkc9102qtrn1<4C^LvKyo<bav6ZT=|<W`#Lx)q4V^ZeXqfAIn?0cfZRv~HlDy*-"
    "1k%x40WY*XGY1k3sLDjVodp17nD9|p<3|>0@UxWJPG{&}0N@m07#A#?PDBxEFbx3TSa3FOX+Ydz=Kzj{Ss66d7Hy&p?LUB49*35}"
    "@iCd1Ae2T6D3S!}H=$FDjf05m7R#0{f}6vXY^9(;TGiaJ@&<~|;xccXMW6)>H01S0Z?12I!fdP}nkseeFcv3dgp6Ao>diDckIKG0"
    "AFiw2-"
    "dxgUNz*^KJU)J2TL8#cdkh{jgut7XT)URG5PGwD+vuMX?v+R0`6IhnL%?mJ&95FLL@Evy(Dj)pCjuBw{Le3=<%%ViZqC31|8kskq"
    "R=aOs4Xt^@XU@kbPozi3Z=7aHGHTOx|O2(FS)NFGSbg5f{`zjGJ{Y=^hoyFL5B4#c0#rp-VVTediSv5NPL2zY{G7b*6je0)m)>rw"
    "162{E2JhLmDAdYVF4RUnekA=C`=-3y<crtkZuxj_htiwR9=d{+eFo(hC?W$Yd{XT3Qg&TJvcymD)<qM!qZl6QAw*3e<+cz)K_u-"
    "&}Y(cd)y?{ij$`z3J;$bzZMkB58lu;C8Fd;7xA#3J8ZG406J=4@EL$Lx96i)Ai9%V_fd5hqW?`pL&*gCp=3y1OGtRPltiPX>@eyT"
    "{iylRD}|VL8tN&ksH7rK+ZT+3XrI(S8Y9{R<|0Gd-DPuvT0l`|!YxZ|$xIt$I`Dj0jYU#2*OS)rxKVr2iiShN%7oKjRRA(LG#Nk$"
    "DUMq(^f87}mevz*WyvjTap<_*EC%OyEDqzX$NP0DNkiD=i8N#RKDSD_*GA1<RuBLHq*a`^qa;4lxdDlkUK9+4q)Qm@yQa77@{Is+"
    "qS-*5rNX&XT-pkNAS6#{lCw~x(?D}pXKCT2!!#vuBY=aKpw%I}2U27=WDG&A3CxP;!p*qku)q<=2FI_Z&X{B129KB<i5neiPCQ6b"
    "dbZWqP+>w-"
    "$s)wmfi#YJ0Gwz4ipV$tyOi`H)t#r66ps#vE}(yZB85!~tT_R)igfq7U$Ko`p@)t{3<jqbA{>;`$gnq@13oj7T}{wc)Lx~M9`@Ub"
    "=<%T~X4lf~nWYt{rPwA)YD%o^88^zZ&s6VLYF#1}t&VBfESm;>TFUV_g{#+zOq+~z7$1&!MDk4ka@(>d#6^cfnohTotoRnl1_NRx"
    "3goxS;>f--%3)AP<{B_*99DGuqu`z}45xL<&`{%%YyIJbM0+txBbv>?nQRRkw^R|%rbJo5W!-"
    "dvLMo(J4HTArL@H{!SAVzCJ_~k{%HXK;Dix!e9d;f8nrMOAK=>f*?w0DFYJt({MEqnsp`-_2x)p|G-bjTzjX&5?&-"
    "%rvqx<LflM+F+4S&r(i8c9<8G5Yd1}|5isn7u@<151>cRo=sG%JA1MriuDK^pRck&TW@1uIpA1825l^)d$x6kNCh0O(*3sGOe3T$"
    "!ywRU^5F>r-Y+xI-LkmfR4UQPW*M9fSo@$IYO{aj<}LwFXW$jZrzk!~+V&<}2A$pFIYwC|-"
    "2}Z4g5Up*F=SG@nQc;dv=o7g3LXAwm$^Pc0b+!{kdSW4H;F*TAK<MLGbUTB0**p$=v`iiA&pkCRi|7_vy-"
    "&){aRI}6%6LEwRI2Fssh^#p4-NP`QSXIR!mfs30LXNR4ysu2LDW9dF_1tmT-<qw(Wtp59?+-"
    "s~o#jIIz7HSY>AD!iy8dzva%N#Q8`&SG&$T6G;1g;i9^`@}DFuGA&C7jNo18sxKA>cud2G9)N%$rd!D@!xm7Xr<SN)Y|wH8P+~zT"
    "H>+q6wN}iN=aJ<=9F)PiZPIuv+8&T9r*Vy(P^*U?JQCz-"
    ">;^rO@2y7I8ZZt~xk5H6&l0g|a2d(^(&U@nB~^>rT*BFIaVdO<)FhEri?jS|Ys*?41B+R##@R6wTA*Jb0MF1%l-"
    "Z8=89I1Q^R5u8@#@S?La&!q6PzfEWdblC){ZQ4oK5_77h0B{HMf_yW9i2=`75XN=5Wu~|V>c2BB!8|^vh7hyP6S}WSvXyR}=ddV>"
    "`U(Yxte4rx!OlvvahJDp8C#@XZrDD5o%TJtihdcv#jEIwokRBJZ4C<*>z7k;IFbfqizS6*hIfdMyW8stXeUcb>Xe{dU<w9c<>Qd-"
    "Qm11_$rX|ur=<iwgP$g0g<NLA7rN!La3wPOvtXDw;A<!!#N)|YQ*o7#nBLpq(uCfJtBfMpFOD#>Y5a2`XKnT~%^vp6V;ADgY4JmY"
    "F=^ZjP&P}fcQi_xnhb(Aj&CKvP7NcJeF~yZ$Y7<epZB8wn$mv*yGen%RZem63<k7HR=6SycG;9Y-"
    "Us!x!yhsH%G<|fNp<T7&Tt7VR><WmtRvbDI=h&R|QocY(XdOTDAn(%O6Q;bkiU2Ze_Vy>Cj#9*};Kak|M9^44?EKQXXiy_U6g#-"
    "Py&u<nPmBFb=ZI~;-"
    "B5<(OtltA<3h7WuKMglaT0fiCC{PAqkFefLegLI$JtuRoFN8o7!~d+WZ*{N3vmuQ&d<EdksWx`l$=r4W5{;k&Rz6KBP|!laF+yzX"
    "5tAALYhts2z?5{0rYlffTNjjuBVb!lQCs8vdABbBZQyK1x{2D`A#C_+^F`;#)>d~+>o|Sg#DzUDW?5EAR8Wy_-"
    "d^DNU9bNqi)0{AhC}VybnwMc*_~3pd1`<A*4*b)KaiQh}5p#PEyQF_SYf-"
    "!N>&<noWP_quubTl0sg2Z5M)9NCG+T=3n=LZ#D;pyTqZc1&@i;sliynxU9Q5e$sD|tg_i8<m-"
    "8xKx(K$mQA}>kt+#qgurIeRr1Url%d;Hl)Aq(J^CpKq<`E~fMYbk06r^~O+3kGB@vZ$s!_241Vlw2RXTXmr|#T~oH2GfNCs~g@3R"
    "tdAaYi=Pe39Bt*REu0g((*RA=)X53L7?fe5f@5JsigzAV0(rbzb8K^h6&FsRSgQ`&sOeajNO@?5wjI2S8jjIc`qY3c4kCZU-"
    "|8_52oH=T?<4mREB{KM3Tb4TrWan!UsX$ZAk?ZD;%OO2DV*}_HtNoiGJ1i1qJCJ2p$n<DKgi?o#vg-rQ}*}4*YxJhwchDn1bs+?M"
    "jW9CN1&rt!M{!uH|CcZ2WECSb}7N{j{rA`hEBzdb@jPBbV1UE#JxI+xSfO*|2z|t-"
    "tx(78u^peCK?C?OTP2zi{rGKvIE~QGJg*&i}AT!_sf<`F*oWcYeJPeI8yP3-"
    "c5Bx^faO_Iwl^a%eAahZjK~R6xh0xn`c|mL<`B1R)@gHykRr}(3@(ymPF(5G`w&ILD<?&#NHR<iI3yseK#ShmG1fQq+u-"
    "=`P$8JAK18PeaJaq4S%DfE-WLg$B?E9A)1pi1j)c|#OLNTD_#6BE+S{Ow%4prQ7c%m>wvR8^DFcpc<uPo<4r_F>O^dqV`CDpV%5i"
    "v^YM-"
    "8R&JsrwU5t4?R%j6paH<)b6!TxTf<gl2NdIT?K<0Fd_@}06N2FynI@sgiUbVP)DQIsHTO_p3_usoJpB6N8OUfl&y87F!FnTkum$#"
    "8)@{xhsaXDCI6=Jn_oVHm%3J8_>+SN*V!g_=`96f|x77Od*9;4Tazzbx|lS~8!G9l1adZ~Mg-1YMnlx4twoF!9-"
    "@L%8ZBF#q7baH7~%CpTTUmJuT~8i(1tHpiBp{=q3#zn$8+Zdc<K9<RIH;^+kqEUp&LlEvQq_(d}5>T=M#hB1vIcl`Wt@Zv#qeZQ>"
    "YIoXzd8E8kg5>0jkiZppbNZCRxj|<3$%W%trP`3gZWMb2I${4oaP_a<?6n5B<{IHIeA-"
    "t3lTLuYE_WfFc_(n`n0P3jkXTFFr2u}xl+ai&@;REz-JP&pyonQsvYUr7FSu!5$zQ}ybj1Od{nQb@mn=Lu+aI02lpzbWsdPP(&%D"
    "Rb{ASLDv)R^!I(GM0|AD_j~<PrMD>9P75%Y@YmNg>uC7}W$_MO;x;1(+5r?uQ_5o+O-w<_6y3PSF}2@{Wet7+jX@EDO>*NLd_LS-"
    "gC{OcQRaaWJMK$_>Of+Yiy}Hd|Ai1rp5`7?r|s7Z{X3(shOK#j(j5o-"
    "UE*@Z)!cBDVH^Gl!e)t}iRL8;ulyJ}syIlhR+?1$dW7X@3y^;4ecrzEtM8a$?6sBK4(qh4g8;kRr^ka52U?1-"
    "y(zV<#sar4~+DWZ)}tTBvBCbB}5>7_u<+3DZD&>ATSF3k#2_nZ24P9mER25%y#$w~?$+7};wh4hs{~_WxmTjY%`b<Us>LQkt=$fy"
    "}!_1x%CK8g@I-"
    "xF?b9WR`57S#nB>?`$#q6gPEeW?uwf_YPRW9(^U!=3CZ_Y6OLOwm^`MfGiGVNm2h@cR;&s(gImPyop1cUN_MoZK1C1Nm(s4OQM&|"
    "X@!?cv&67YT^gIP&w^Z)F3o@>v_%zSUp5<QsKM=fw9$iGgyrPoE$dR!%JH)+G-"
    ")2@t@UvE!VbZcM*PcNOtMYc47zq$|EZ=M3qs(r4Na+LKpa1p!R_a<=tLfsI4@@2Y82SGtNY9=)DL|fsHZJcd#j_FmQfE&_mwhJcM"
    "vJVvG+eM>qNVWn^(&@8ZJF_cZaJj+_Tcl3%hHrN|W$bI7`qdq?@3nIk3v>S(os6)MTvRYziI=^M}J?5+Z5ZZVb_J)hfo^wtpZtH{"
    "};bp{_0Y@CijOIx<0qY+>EgNcCAWaD!td_IV{Xp<4kqInnm5%u*T-"
    "#TMYdYgtWM?4i)Z;8w*%dOj}YC$Lp1h!Hl#51U<94#a2t(Gmo}<I|>l&?l1E7ZP+NKj4OG`OEWo^jwi6yrOkcZSp`q8)5!v+t^D|"
    "Pj@hk1n@4%)pFu@UKM%pbm+qlEP-"
    ">y(Z_(8|DcBl>O~ICabeHx22nQR`81D|<oFiT1#c%7mrL5zjkvjM^G~Nyv*1?xK;#=Rz6t>h7HqTQTy*BKk%Lv*!D0hon&f&ga_|"
    "G`r^IuhbZj}W-"
    "cXpj*oEAMRcu9<2XV(b$=1sjlhzs>B{~hQ<Ny<AQ|NZq1D;c<rN?UcEioV?>4KaVtfO<283?4!R0V7VX%T0?2gw)CV#6AoWCYtk1"
    "$X)L{c+ajnA;eN!O9#df2MZFDq_xrV{l)==LB0CDE*-wu^53)3H`ZhD_Asyvjkt0mhaFvE-*7TqZMnhc%&+ZH~M~N(g4-"
    "lRFh?8R}kO>J-#>}!_>*H-"
    "<vk?j;EB(suqdYwsB^`JG+HY2<b_Hp&V9Zn&cpvCD4CG&AO;K321*vB!KYRgG87xsX=6i^aP*NAEyknK&-"
    "+c=G<&K!mW5<qwvtm_sJ?Zs@X#;pj?qbXF%O4YFbgdfk9xUkX004b<=x=0d?#_l&A?i6dE=~&dw<lG#u%OJfd3sA}%}-"
    "3yeyPtf4Vdg40$;I>GH5C`oA!LRv~#3;dW0=JVWuIEK05FRXKC0-"
    "PJmU*0~r{mK<++ox+^A;<`!!vdfOTSjNf_XP!G5_drTKbWAYJ?YLz+lG9a%=_+R7+)0vL^b1~RYN6uPMA|mg9MV0jw^X9Bou>Nrl"
    "{<4h%qI}PCLBNo+en+3y_f`_D(BRBHdQS0m#U&yjihz<sia6_4_5X+f9V$UTU0h+=Akw<#@NIqS(}ucI26!<4{6^%?y$5(*>9HpZ"
    "zDPw2hwkC}J*2H+~u%{Yiqb-"
    ">jY)kW;@w^|kXL>=u{+v!{UdIBT>0Q&|)Y8(l7ph$#$z$Uo#ZU~z)dJ(>REGEYh8ZdmkckT$R(LEw^3e_$oBBb>{%3XaHRVaidtB"
    "s2^k4Y!MJtS(|O3bdy)g+sZZHxDHMZ~<4U9rau|QXL|$iGAWdhBlI(V&(k!1CWVg4$)J?rmDiW@}T*Bzf&fM<{m*dzi6D}ZgAY4w"
    "yXqVNco*nX7boqWf9@9*(ZrdpAZB)!nF}n(Vhq&99g5+sH@B06rjjn$sA-QC1-"
    "<L0tC(F`)aPxv+oOJ;|gt)9OP0F<I<6RRc0*M#2D=7WqDu?0(<0zGPyYwvPt271#m;XKspV_{(WU=M_s=lRnwxRjvo2DmZBbe#{G"
    "1M(3ZZxW5UIo@K~JFeS(+G?sG~3qwSA5)MxgPGay5_f6mhDEKlowhvW_7!+^#H&{7j2-m$!Gn1<e~AvH~Rxd2)~2{}Z^YlwrFF-"
    "p6v#*xRMje!#xn#+k@JP(j?IC4u_A?__UP8!LRj8nNHGofu(Ys+2I1*EgEW=c<5<(nY89~{QaBaly`QGJND_ADhX$uXtEvVUsiei"
    ")mfx3*(d>NxSqCjUpC-8e={lz}^>U4_QZ@)P9;>J-"
    "eU*D1j;;hGoCpE${bcuk&XmhcQQ3ppODYp?`JlvyJI1YJ{WL1<GxWod!6+nxm!^(?m9)IGeEvhLe%m781*-"
    "6BA*;{i?PE3~p~RrWM&UGup|n8*OF4%GwCik#qM+RYa`UV8m6%Zz9HWgU-1&0jG@NW368KbV%X0IoT})5}f=rvjccw24-"
    "{#^yN?7_b1T-fD26(B3u>KD*4_O(lxpV_CZQ1X>Op8=9r$mHRYX4Zm$ezI1%D+QnjL93_i1O=bvV&q-"
    "A@(l{tSBSfBN6blnb!@dGGvC1PKIUMMctr=6PfyK#ZJDY80&tFr+wrq*cMSUr7acAq2q=ou1*RRJaX=BiZ2-"
    "=DA21Gtg61ha|<VD~o)@%fYt!{)PZU?irhh<7sSQ9Tyf+gE3%}%JV7H2VVZv=IQo*{wZ<+RcvyfAFWJI?uS`U6nd!xJ=&U1DCn$L"
    "T7cEHtJ2B>DDEf__=GQNy-"
    "*+MxDSpP%U&u(PhUDglq$A_>0=zo7A>#(?uJv+*j`3D*Ijb;t($q?m;Yn{tote)_q?eOvSd0IOe)aq^S!@Cgw7Rs++JJ#qVZY?c)"
    "iB3`Pf(D~2O$?DTVk_K6POSuIuh_<i+U@9@9M*%(4*wb<Z>*c>U$E>A(p(ffGbY^7wY=H^l@v~5Y{=rB(O7xM=7z1Gf;}ICYp=Ug"
    "c)#GDodYoT&tmqYCk~+fEES3SKAEW4?ZRDsm(1+$O82Xu)BZ6)$DI!s*@_`EQ#l@<Jq{e8O(P;k3v+i25x($HAoSXD=lNDmQ%L0H"
    "yBNNCEC{V**)BbN6=xjNp6fu04gM@`S%`$N*cL)tOf;fN_^~p~%1)C6(z*)soS)Mf0`SNKyi8R$kSAcH;RQaNG53KPpT?aKxKpCU"
    "n$K%13<Vy*eUFpexXj4~~^N~d>A)R()&m!tG1%ab=7hel`b4&~4l$#}1(?ZcoY;Kl}nu&A4ydh=a)^o}g|DfN2_ze<Jas>!m+>I9"
    "6EYmWR066v-m6$Ydh@r@X+03m5bQ#<toMWe#EmJ$J0+`x<{O2u8Hw2e5Xeu?=kNzhJ0(&x#7+8qdd?7kx-"
    "L7JMUFfpS{uwvvsgyI#y`%jbpLT3%PqA7)#+`<vfeE*0O7n`U=d<Bv%_6`Qc6t;isB6LvKrZIeAWHg@dJ=R<kBa!(Z@vov=;ygar"
    "4L7AV?6<;f;4LjN?lYefYF4Zd_Zhbj;-tNLJ<xSb0Z$axwY)Yni<;6X>4}Oi+zug`|(kN2}xv)7~D-"
    "4ujIE$a`EaN#~}Gn9fO@xNBKyFm$QR>s+%pU!;oPGiQ}19VECsPuHa*cGVy-"
    "$?60q))^ZCJT=k{)oTCpLj0pwiGhH)U+zJ%1d^lmUkn?r~GXTNeWe7q7F+j7Q`-"
    "}JICo^q3I7dq@Jw)<{mInIUPM#oTHQlibtDrQ1kno;c(CD+HXlu4af$iz}W=XK)P$R<x1MWeGQ|v-"
    "mNV4Ur0=eA~#ZCGZHd8At4?c-JU*@{)TNn;#6NosB(VMecK{p|KO-RGI;8g<)w{WHkbogd9h>cF-G*yA|XFEL72xy-tJ+@>aqF?o"
    "p2NxqVr{~fAWVi*DY>;5#lThETXL?$$2WV<o^TK#MWy^rXibYSW-"
    "{g(D(HG4c%eGg|xuy3(+I*qmg=yoa1BBcQ#Zx|!L&uZWA091{;hN}**u;<E{_GLEMFdE(6Zt@+&|(0r2`K&%jX(0q2KPH$m+@Mn>"
    "EpE@J!Y^l8v<VYc*Vz&($jaf<e)72XmLOVxTbO3gyy*hwlu=~FAwR7GtBXXlUC;NfFMj-"
    "f;P3vO$V~62dSaXs@Z~^_RlVT4BYxXaXK+@qM+5qqqU(mH;L3SyNJ+6!{nC?J{eTYmfb-"
    "8%&{6tPQ4&+oD2kv2s0d?wj`t7w5x~M$>M4Gz&bstabOB5sV9(0wtT>ZmItOFmn!4oX59yX04GBIB8gHY-cf@x)CJ!)Pxh&aG@!N"
    "|=lCA?;+Vne#l3ynoce~28M`n*mJLmO=P*-"
    "68|WB3b#mf}k34iZww|(a(!P456!~oX7>#jri5NS0D1T?!13?)DTjnTPKg}2M2k`>SD1Y<$9L5rxK$x~C7{f8ri*dqv95UaEJ!5n"
    "|p<sq?03rkUVPPNX9A>7_g>WiMgCEA!dT=CV?kbCF4om0O{Oc<4v8*JiXS6~d&y-"
    "n^udm~9@EIX{TTKMOd$U64*ujt%c`P;$Ni?*^bTa2<@0u0`P^<-"
    "sKJ;gSi5vW&m;aS_Is)uznU3izInLp;KM&>O3`;DOUbr@bBSUw+sSt+*cA00;s{2eI5b$j~t$7j7EL9}F(MnEPee*?pG1~6kpQw1"
    "7Eblojt7fReaq%!bSBkl&Z2Sy^)5!R%No`mFOha5NClXkO0eOVpHt1IvFn@*$XiF}bf>}<`vB!j&341%;UBL(&KFP$LdcYu*bijG"
    "Wwc)vNjwWFAntTmPVn-"
    "(?<D^yArsQz1>!mw97Q)blXpOYABRF9bt{)=;%1JQO>=>qD!g+2bqG!|%fCx;YqIEGxn(c`Z!ud05;`5v`TyDQ#iq>o{MteN95!M"
    "cY?$c9Gwcn;;d>w?*_U18pG3DwopV&81Yc;p~nK}#n(vzVMHh^#j7HOvR7#~wCb<09d0<PWu^PQ9Frqdp)medF$!j~$g>%8H!faG"
    "W0%1BY-k5PrpOOJ<4JX>sJLLFfbY}VCTR@^wzV+hAxmNB7OdCh|zJ*fD{Ps6vCmcjfv7~8QAs?)W-Z=7lP-"
    "*F<*KP+!{#ysJ|L2dP&GGpNxVmgD|J7yF^Tcq2jtS-(?=)I_5lzvbz{m?iks3zd!ZT-@;pY|VhB4*d+`)2<^Kub=L(-"
    "WjXh=7aXv<j)bAT2m_q4@p0Ao;1`xxb@M+t7r=ec5^-"
    "7R!BZoxswfmZ%b*J;Z!LPT4d}xOn1S`~hq@UpiSkf;~5c%Q2Ct`5Ym!R!DX`LL02m(DAn7rX`!lj<&EC_YX*gd*|udW2D77ez!0@"
    "dDz*~p#{r4a?fB$Ny5hN+JYP6U@nAXxjO|Vm3DeDLH9&*4Bu_{npTR;XqfL<!!VqMb`KV*B6WEnJMR;}`=seGvx0L9%p{%2&A39-"
    "O-"
    "M}Q$aFLGv)5MbB@WZ${_*EAvQaQIcngQZSCb?@&zqx<<B>&a96{O9WAb*dLWj9D7?ZWQ$82n&#V6Z^GL(>ln=;U#fE0ReMV>un7g"
    "TM9w_XfVwA(oSEDuU5J6QB}%pgIK{%VS-O^ac=)5xn4DHvU0bZsR|oDLRbe)j(UzMPL{lVZT36D-"
    "7>7>k~iOq~u~*0LLd`A1GpF<M2U5BJC2!)r^D3AQ+_onw3c{OR!YQB#tMp}G8M$2|}5MIutd14YRb5?veRMGnQiJhR0Z0+cTah7t"
    "@e>da6;;o-DQBLu_I^32)|UbvzbA_qOGqSGKIgaQd^EBgSHUDtU6jHQ_`8MzbspYQat{Q`%EasxQR$Ch6_3gb<83tOp-"
    "M;D{IWpgC%BC;mhi4d*AMltJNENq*RtS?=um^kWb=TMwaY=MSVb@O^#rHrB<{h3o`?MB6A26BRS;7h)LI0=OdM-9w@nmPrI-"
    "`!`J0|j&AVcXDS*i<EL6pl5;mb3SA5%J_%coEUDPY)8}H@=OWve^N&X+I3^PGj;U+Uvn0n7>@eQDs!sY=noj`~=Fv#HM4P--gR+r"
    "pUub87lQMl_FajP6-dX^p>i59tObByu}(grjj9b*@4l7BLp6YM6C9HN-"
    "t>VG=@<wzGz;Uu<JBtGIKhTMQ|&k5m(wWV~6JrH`k;3W}ZCZ82izTLZX^t9C9FzZB)9s`54AUD-"
    "~J&%wEiJ$p?E1V#O)15S)vMY%ptAxe@*GtNr2{DKSk3!G0;npfURn6_OhYZ{0$HAl6dkSR@%O63FH@`9%6wNG56XXX~t!$(4mi!H"
    "2?DJ?F+#(lt+-BwH!C#uMW?NoZ;3EY6l*joPMRn;PwK>^Gf!)7(8d($_sp29z_|AXXHUG&wGzGL^+4rH4oRK*7o5j!|U{-"
    "eA{PWw0%3fC+|E&?^IVPjAJ^936Oh7B8a;6ip-eyR@2MyQk_jX&)vY!9(%sky;SiCWwF;dsN;Z;o65%vw6fy;-"
    "Y{_8*>;v=nN}Xwp{=@dutadbNl_WOWv1#NK+t>p}XY2xlIKi{K&IfWypY!wDMU(k#ytCdCUVN!_VS(e&$m=(&qC^u(9ML^`yDyaS"
    "OSP?ZWCsjtu>WtHO-&0G@gNfK<=8!}UzroX>|Plc5pB1}wbXCq~1ed||^>&SIF=c>)+<UgpKL9_%Ac1V}Irx3Fg81s-"
    "CY&{VRFzQWn~F+U>lD5^m^2q^1%jeNF^3uEtTOy=Z?f!_&+<{-"
    "BxZRxkA8^Fr_M&Afu941JySO+IYYJTZ!(TT$0vK}<{<<Bl|9;R6N!Mm1pVpX+wly6#>HawyXwcC7Tw<wY<Mx!s3g(mJ9b|^`-"
    "IdCPWu6PhMAny3NkV6|lo{BwgJ|&togj?(fdksst9(GC5%t9NrY*BJh%>2e<L~;mdhMhui|4CzDj}DkTL3PIJ%t9O(tV$b;;JrN~"
    "p5Z&td9zt_En|UtcFVo{F_aL8tj|p2QoPc>#IhC=Sim#v6gN334{Htq7^7z!LV8SJ02e2pvFB0ZL+CNiXxN8q<}h&v%z<WOMVmyw"
    "hcgjfGcdv$mqBlxUOELyPYluM1DlhBd?cQY!5g-P_JXfn?GYR90ny)>$;pQ-Wc+CPiAS3DJZOUq=9H)0pyx#$j3L2IO1~#K0dmpc"
    "U*8@nNW6T$sYK;&jOieZWoG;B`Kq@Iiyf8d3gHs91_Q=DNqv;!Y;ay%yX+QtF&LQO)GYE$<zHhy3v(5?(O#qLd3A&dOuVt+9c8u;"
    "DIb4)<`%{9vu^Jh0GKwbkw&Hqga}X(*cUEGnA!1g%G;6TwHX|`w)0IK2=8Z<(Dd(favG`fGWhSO7GDh3;E^;h@i0^K7rF7Rk;S-"
    "C;XR1>sEwc_!Ocb}iKX!Ub%#8AmVv7{(%h0SVP*`Ad2AyNuZ%oHFWjBQp0aMXyW@7=wpTp~^mfzCew+v5L~6z?7<|}^!0d^$fQ4`"
    "qb^@xfz%fYSTuQS@Qd$Bq#RS60Qcyf|-"
    "FfHWL4~NtL><#<Jo7aIb;pFCt%t#2e37N8HfqJ=bf$?n#4OewpEF*X(E<xYW~7N%)IsFA7|JfASVhs&Q%B@Yt8$mivj>=4q{mqi@"
    "KVRglJbZ)RZTc%fHIl4Z8k^ETr^2&S@ZZa5MNkc;ADXAIad}d!_#d&3pWk-"
    "P`BFtpT2sILbeRq0}!rRCdJEHo>4)!WtFQDJu(~)$Ffi*{p(RG!L6Z}-"
    "?NUIw(%nU9e7XW6TTh|4W@H`)6w5BzxbM~lC2EJtai)H=~vZ^H3)3TNIQ#k{nm^p5_?)<wWgs1!br5$;ks#xZ`&7+R#<{Hzy|4~r"
    "~J{0&rztvb}u4(vVW<6!I{O|5NEa`4R88^-6~>hrsE{0>k0Tl_L-xInhi9aFke%=leE*ES5qIIYU+1dFZ96I{y8XPL}2qG_jGH7E"
    "J8<Oc+s+x@VT6JUweG!6f?Aa1n=X1SEP^FL$ERcSWm75M`-$07Nq4(+Se1R=Cwb0z~ofE!wf`r!K`RT(~Y3MVUc979rh51-"
    ")9K4r;)E-"
    "mRr0B+&aSt9!?D8L@@sQp|v1BaQ7}Vp3jVeiJ=cxV&9kCI)<`AFdI&$ifVwz4okV+GMP8vLWqUBGm1_R&h$9(8#3iOJ!u&ak1nAv"
    "T8IQyy+^mgb*SVScdO@K35u+o0vMA{JW(ZOqCT9b?RYU<juX|bF;*NSP+!_Q*zbvH9JnC%f^MPILU!>;ya@Yn41nnEWRhJT_;(bU"
    "&$G&mY~oG&#WZxFjuLw}hhlhY+(a;c%}d%AIOPG2a0c@(Y*CoHmid@bfMtc?OK)0&vB}%`knp4elLhamgT_%-"
    "w!^%0G7}3|X2NF{xL<4BdF@G(CkueqjV<5{TDW2*QyR%xR!vjHl@|XASZlkECU-"
    "j*KoX6a;weBP>;5Sd&_|gv5UI8u>08aq1P;K`ebLt)tNmQ_V8+>J`sGvt6BP+Sv^cX^M~RLjy0P#b`RxMzB<$#^?=0V3e*&U#jfK"
    "5I+Hu|n{JC<>^z&R(JolMo7(FxC`Ic6Z9A`MdYyd#2xdtk28D|I+{zy^3WK95o3?}v<V7jdp*ZT&!5_hiT0*tV3B;@aM8RK&hD}A"
    "s=R&>~h;cB!wu)lk*)N{g8ojLBs!$j^!8m(h(XDq_;lTi%#!GkB}`Lh^IVyVsYMB4k04B>_9K%S~a^-"
    "up$#+<xK=q3z1>|bKKwjLPXFExk~S2{!96h0{fv(|qR@_8PmT0xvgTjinNI@z?P<%T_=IqfAL4g)T53oO2}1EBRi3=fjWmDfSH%M"
    "3Qfh^%XX7Dk!O)b!wl3Zq^F22qRkNh|KTh1^CF+8H^D3KfQLcPr|zIp1g=PohQWIFe7vxNhQ|{nxS|3pbe+M=JqAZE~4CJppq_t%"
    "k9al5;R?#Ml|S$12SlL^ImamhO};m<$DXIK**jtj`y54X_R~!@%90YX>$NKXr{Q!x5-"
    "FS35s@Z=_($93ar)sGgA5ApL_OT{fu%E_#;RV$AB^Y#RI+A09gl1jxsjdJK#3QeVk9{7m`5q;8nlG18C*(Tz2GZR}rlJ3NYsDSrV"
    "629bqoZPE+N0;osF5;`=r=r5UsEC!Sy2<u0gE<RV=2G+lso_9bui0)|rz4!AngsnYr;YtHfd>xUx685t}gbEE?cj~mCu@w-"
    "zImSk+Q!=JIChH-A%qnXlmL)Cn%;ab<Vp#-"
    "cPO6(^ML63~P={L{xhLq{!?}EPrttTPfga6WH9K|v*%ThTwe6_8GuMKlX1M^k)=f)DTRcOEw--cCSZnljeQ3eEi;d3-"
    "^&Q?e8E1`1?e=wsOVBXE$(#&MTf@@><K*0sbh0ef``mO*189r+No#6`H<LW^9D=EU>M+q797G{O%waGfCv7f%#_)2z@`R$%6v1dT"
    "T3=on*yNDW;1f71@dXka%s&LlC?<5|;O$Q(>fMa^4tnaO?97Jgh)D?Bg5WZ*z*9Wr>c}HzO-"
    "y0gbx*`7_>3|Xx+_ysM*Wl+Yn?0>ML(}-%Z{H%Xr|F1iwNWo@BkuL)YTYB+OTGz_}Epa@#W1-EDZY-U56)$-"
    "LkgDs%hag2<$Y4+d$32CJX#5Z<)VQl8>eo2^!FDE*QjW49h~kl$~abF+HLjZ9@&MrqG3``8gVt=$|$#8RK=%kxWU_*yqyc2Yf-wc"
    "uYTnYFz#ghr*W14?$B=H~R>}4@59nqB)ig)U$bT&`5W1PFvvn=~<oaHyD5pH3q#{o93t^peCyJXE(px^&nLq0k^Z!+^qef)`#2k)"
    "%kqr5*?H5SAm1egI;ZLIKT~}jk*|OzfG6gHI>}f?-"
    "x=Q`R>ShP;iD&q%lJ#0@$UjLrA$tQk5`bIhL6MLe}7x;Zh5w5XP#>A);f);w4SFw>(XVXDud*GbE;{keJgz4PtGU&W^bd&*nVIL}"
    "cuNQ!W@rUt;0iZY0B<0(g@veXB6*_~_+*>8}H0r9qx)E)T1H!Gj<l!`UOt2G6h?wm^vKW}m|>=Rb!Q6P?Q0&zg&QnVFZe7;_?<C;"
    "Q?rbQ59ctfxI@L;1>{4wPWFJ(8WBJfzReQzp1Fqw|$E7^`rTz411FKXs(hPiQspx^fx751zK7`Dz$Eyiu%!@lZ^Y3TCU*HRL7LG+"
    "e4I3}46-5yHfj4XicTx5%J><FN-EU%bAo(@H5VCLSPij0u{#Ecs4~7<6qo@l*<pIGi5xs=;}-"
    "z9e_)9&5O&zr>>@f3k^Yj_R`A9`w)hEr4_QQ8_ISM_Iwknue)uI@_|VVABgcCuzNfY7nar)Q}5zDGYc*B>}_g^T@I^-"
    "$(U@rVIPlS$v#_SDn52m<cf=pwj+g@&NkdmP34L)<L>}A9vER*o5+JdZZVBnZS3t<0vgZt5I_%iep3f2)jJq@L1?h7m-"
    "3{d5lXWq)dnfq4UBJ4+;Sdy>@drA3g4EgO*#<!3ZJPhTK1^IUL-Y8=W>3k4_-qoiX9*bT~YzP~{@<x4B1h>Y|z7EXSE?-"
    "HfvjGmP+xDh1%`?p={`!}}=4D}CDwOT(iV^GwINzOlSJ!qHh8PqHtJpD`&<-06Z^*tTsJYu$sVaHwuIk~}R6xw!{*g?j1K1-"
    "#F`j=1%tVL3`5NWBb1RYWQ6#ijF0FsJ982;}GauJHD+53_W9fnj0et3FHH=b&VIKgRm$)T_I1K1i@J|1x*MUZ@WbCLB3^6^nPJIo"
    "+J7Y%;@@g<lWC@~w0su@Iie;hCaADX?mjoR{h(iJiyocBt_K!dEe=Cr4|H7r~%4ySnB};JtZMu4Roc*axc47JMG;GI8w0@tY-"
    "^+B1#|&W`4>YDJ7uF!hklSFa5FL+Dt9u~CD`Jn$Uu*5Zj)<A8D<&d}*U8&W1mAgCYS7B%4tdIP2zEU+qj;qZ|bZ%3>iRkP=14G$M"
    "tqljzhKk=R$cDdH6lII&QjqTWM(eWFPj<*dD&b4A*)-"
    "!LJfC9lI!FAlS$faygw5tD%nT01mGCIANqx!=d|2X%rUJ7$8JTs&@@v5(foaccn^e<u+TjpS?8B(<`Gy2E8-"
    "EKk4D$U`<eq0N7dh}%t7lbkehma0l@%PbmGS1xaRn}irrygZS8<fb99exb<DEc=M`%AwckO?jdlhkreAZ3J>PiK1N8iA$K-"
    "4Qb=oz=VtJUr8(XqaSlcC!qr#Ox)XqQYSg2(4O@l+=T->#2}fskHQ!1P9@rQ6shQH*E#+H+p-I6#MZlA^!m4(-"
    "<#_2=t%l(KoosMjlJ#F-"
    "q<ZdCm+(XaIr)YdeZZL*pK(ibki{KZlqN7M3HCJrx*lV4$be%fPBfT0`E(J4tML*j<<r%I!Clnk9_Kk|!-"
    "z#S9SosB+&ezbIE^Xfe>7B}JGL4PP5(v(NM*AisGKXyNI6r8L5F2pk+KIV{uC<UqaqJV-nzH$eS{J2a4^&z<Hi)4qfuYO1l3-"
    "PApZtNUeg$nBWE$*Hh2MJLbT3n*pttYJVpBNEZz+LkObpYBnnxO_k>D>@#@2}6Cfc;2sj+poP3M+`7ZSkH1RM99YhFx{=`l~41te"
    ")Ej*C4t3r-"
    "9o6Io@EA5zna_THCPJM5Jp0JT5A?zms9`@4&g*8oheMZ`+>Zlo`A@<Wf?URQzVmFKrZSj?O0OXTIiUoQ`_toryopR&e>(PUb2)>d"
    "Qj{<k34#RzT89ucH>V|$IhmnHmxtM0wDm=j4&N*gkg`}BD#$aVCu$ida9M{vd`MQUZh+03GtlTsvIbep%IWKe4(<WP+ry9I52byz"
    "+@SI_5$2Fgp&@^^<W1EO{O}NxNxXWz$MF@!478KS!|rnq&`rg;6(<M`XNy8vFUR+k4Vx$WP`}gd$o_)2zITd0iD(@AKnx+4gm*R&"
    "e>TM;Wn^E?3J(rEk!T^?El;P>5Sd_g@1HLl{a{AH9MZHSMW7#P`vv?>zdso^pyB=+PHZ}e1PW&2DwcsRb($Y98`6h4xqE?R)T`g="
    "1ypf)97tMt{!&qwY;QAfFJIYrh4X>40QJZ{_eM~vW|mM!yFVkH+!a4CB90NwjKEP$+5CzM|KQ>k)29ok{1kC7#fy`ffHFs+KeQv*"
    "kn~Ur9ItMcm&CE*q$N*o5Jg;+&gI@K<HsqVnGHV*VryGDlvV&sMLU!J;W2&k>FF1Kw(=zeK%{uh+;o&CG$j*P~`ch$0qypmlzS3b"
    "^Xk*?`brFDKlj+?!Mah?WZI61OixyH7*{W0uprGh1w{}7#JQyaE-UNUz*3v0l*d)X}|UnaQCq2eZ)YJZ2~t>WUQBt@n0AhvtEhd0"
    "iyB*Fv^6oo2*_owmE({3Ir}@8KEP~&6#1J=RET@OhwyeWr686YEGzTU;ZX-"
    "O8^O$)J~x9(Gg^XCN4gy)de4c4LX7(VgA+2SO)VeC^pXR-AZ~;XmTtMK?W@*)6uFrV$=I65=fGEW4qXfETYDFhArvm!kQhK^LL=W"
    "U`7jZs{O_hx}Rm+*VfXTf_?_$6<GL5_c6zoGYiFj^d6965Xb3HWj={hB^%vmbw*KWu=tF^*U8k2-"
    "g<*S;;!nC5N>xT!{<0rWHX)NnDjSjKGbw0#>(yP7Tx$YS3KT+GN|DGYzXPyMhg=hBE1$}XI#TIvz2|z4LcQ_jq;X}#-"
    "I!Y)#1a7;X;+m8)kb%t}|j73B<XQa+zqW*UBWLmrij#v{-"
    "oB0Oj0De~@lNI5ejyJ1~@IcI2I3O+*5~r;S0rFn(qe%u~00xI<kEKv*=Giglo>%$?ZwKo-"
    "b8URlNS`go1QGQI3{BEdIS&jZZ485Z#1{yeM;Pf!N3I(8Ej@;N}+q$$W|6ob1a)O*AePF;3dDNllFp^^`lT=cejG808nVHUdk;0x"
    "kDv`Fm=i-4*AAj#RlESoBr&f|cB9YR{tGo72+ZO~XwI+WwuP8S{3MN1EphCvuKZdgHNOxeEIUE-"
    ">W7P&NN%yt?+=4>{1tJ9Ak_V74wkexVx$g{fhU?#KYk;cjO#XUJk&Cm>m+LPyyqV&}NdLSM<3s=frziG!le?Rk>vvz@KZMC6Wu1T"
    "etPO%iGSh4x$&$A1orSPXfGQVF&G24RSmCq^&4K40)I80Tk!Fqztu^n<b&JxYR1+E9~mev3iu_J){>`I&kA<6Q#*1VS<1z-"
    "UKO^Y91-Iud3dNnwJ^lB%u7{ZQ|a8tC9V}ltCd!chx9Du?$TxWtalqS!<vow?}2TW%x-"
    "6@C;Vj=u6^8@G@xOzXHju4a{*w_XcjAi72VO$T1-"
    "Ea$l<D&kw^55=o{eY?Eq_7wL_f0ZW3J!WbF^P+2N|u%E+8~5Og)Fl!U<O8hW1WA?5rp{Z%mR&Xxp3nB^~|~`+H11~4(^B8!6sHVV"
    "lFAwW&h3AA;j2wm*FZ&?_wzK==P*TS(uN>vnjF2xqmmW56vC3nU;mEQK>ewVy?&UDoE5cj2H+H5K6yhmTk6)P{YoHYouXNBWl<|$"
    "HU`-KMm*y(+K`)qVS;87GdF5`ckBeA7xBm#bbD{YA5qd(El4EFEt*#Oq_dYlvSV2uIkVi-F-iG;zIYsZ;16aQq2jYq1zv!oH-C_G"
    "G!M25<pKONp9xQOJYXcPmDLP{D!`T^yoKgmD6#1Ai>DL#~+pT1YzPPSCEm*_=D0=?_oK?KTHn}wk=q%Sj@cXZBa3^qN73`?zb)*4"
    "ELNef9#U6Jh1&okbmxUP%RM-;%MRKfIN}a{-"
    "#&pY{=wED5JYjWzNrLE*J>%mwAabn2plSa>_D3xRw{#p^jcs>fKQOlE;0OD}Kfupl@t1vYg)J2{X7<!CA<F0mZ|^o2K={Ik7}*9T"
    "pDlq~ZK7N5$b2`y1euYq7hhwym@;!T@_3PlE$fg?R!AHct1i0mshIUV^z4k9T5$SfM_bOo2HBH2G^dC<GW}F3$tuieHWUK}XJg$%"
    "eBuL|JS{Eu6YuvROU#yL;3X6xbYM0srx`ZT>s~+nL&Ci>|`?X^X=0Znlb{%{)iz6I_G-"
    "2G|hO+@O2GL?F+wUniVYngc;V!JwGj7$5~4C<C)CfH6V2Wh-v4qG1!0(|-"
    "~cCP8jy_YL^n)n^?Jo(luL>>yl!fw_jjDZwJg#;;r2a!#^ddEmo5*hz=$Cx1bE+m2(!N`csBDk3L6^s$@$l?CCJX{}hh&uH!7Sd3"
    "{7zKm2I;hl~@3BzyTHvpQHtcK>$_26mK^Rxaw&?Y#ZAqEsc4$;Tp?Tv}_NeE&^xZW)dw+VANt&(hZ+t=~ZnTKI_%e8+fGvxs}OAx"
    "<moC=s&LlD7{o9+b*mh13%`%&Hppr6)ot<aZe#!Nkq>!oVC{xc`Mws0FBA(&&`ZL0_h3HxIt5<)~YO-"
    "`8u(v!%_bZZQthM<Jhv|OA!OdFaLIHkP6A+44n^wXG0(d;32Wi&OBKHf(7-NNhsem>jIwlUS9t@CBt+YB*z)h@Uq8Gh~c>-"
    "$D7pCShYB^b5%;gx(+i(9YGrYrn74!L1zh=&<tlpLwZwq3t{P<N&m9B5d;50osgMlM^qvMh~-@bhO-"
    "E<>@YKlBvJkO}~aeGJDSZFF$C^UFj3@ztbTf`zxt3x;vh2r#llgJK}$r(xkW!Vfs_H?JM*3=D$+{eRTvf%*tJ_+jQ#v}!Dvry&P-"
    "cdwFygzNyRUCpq7Vc1LxYU}@CTrIb$vHb|>Z6S_?9LoytMKq-jr!~?tzeszX&x3VP;`VeU?(XK0UHdI1ZbNN?nBWjWvZW1{2(<#S"
    "d5VM(#x^+WmjR09pyteR+Y)XhARS3M=_n+)goQuPjRmKjNJ5k2flP{>fvWzbtexR+hkXwbIGq`{nHY|Wr{};pCI*pep&G~$?9dGl"
    "gUA^Gm<CH)1I%O@)+drV<bloXKU=wR>ZVNuZ>)PzG&?a{01R-ql@65qWI5Ypnv=-"
    "`YBQa}at9ttNAVPVEwUgwdSr+@MG2dcN`PL#%t-+zvw?)gEO~Td(NLt>1z^tjzf|(EkQiqG1qXV6P8~)2h>(dy#lSvP-"
    "ydk3WtiLT816qH+6pdu=VTX9Uf-zxCcSj5SJOW#vt{r#i^T!P^O&<N5k!ndmEK*<Yqwwbw8dtTaH;-WZk<F`aQL5|Z-"
    "=#iGB3YoqZi_2@TDxUK<5r>Gm#sPQTQDFa|iofgHgBHLOe<OPdo8odyJtLC9}*(izm<_$?`nBlpZtH=2)lTS;M7dvIrbtaFtBR7M"
    "3h!=0xpSKn>K4SoL6phvSA}F<up^<w&c3+^jhXkjReKN_zvJcX(cJ&PPV#i^bCN=(X6J%w{#UihxnReYiEzaJ(FOCMJe$QnMb2Hd"
    "`A>OJpqOatZ;WoGlw5MeFm(+jlWw>BP}bU@-"
    "8@9YR21)dLaqniNdQqDqqzBXM6u%MZdGZVhP@VGm353lMCrgvOs6Mz~3iO0$?*F+<p`wGev~OYMcq<Azi(Iu_Wqk{Mj~&6oQfy(l"
    "GhvyW+B4*7t`GT{nwf9@9OGjUH!4?>s($36=IlBHmPVThdDlvh9VB6gdd%FfxDMj}kYIOmV2vPX)5s$;y8-"
    "5Z4Wj78r#AV8=Nkc9!g$e1zy*i}3cCr}4j-"
    "?D4wA?B~ZHr<Bfk@I0OK;FnJe`#9j0h@HKVbcak`%xaz#&()MN|+6+g47O~bk6r7`#Q)2_Q)P7K6~W3@HL4WSq3*69bjIV(Z!@Wg"
    "(R>8i}g;;?LiMiH;L~rH9Itfh2<!-31bWg#j1_f!If^JzMk$jZE<X~{QE~Q4HmxH?m31OZU)?lVR3j#iqPR+AXnzihIL}|6g-"
    "nf9?qDB>hi_5q+8Wk<Dconkq^9{%sASkNVASdl;%+b1(zNk31KY(V(aqEDLTf#%+=W!1@|9%L5un54Zq05VxTD_+B-"
    "PJ<wRjTnLJ6^pm%we&?km$#?|f`CPmQ=n4vNq_Doq~Vg1UoU!LR}d_*R8uv<+{OjsTp4O2@0FlX?x!`7G*h~Y?aI}hVK6#Dz-"
    "m3wq12s4Bv!m7Uo19y<?YwNdAkTBOBTSnSGUl};!ZNxm*_j?qOOyRd6(J6l=P%<)lhUxa3uiEtZj=5U+XXc~?XAmc8gIV|9u%2|g"
    "n}KeLV%}rC<pbe_NRgYwQ}&i^?Pg*t=yIWdJ#!7K%v(lgKF@lnHe48s{XVn?2Vud(%I=;}9Y7oWl+Tn_`p{Y8lP!Zw+u$T}Yx5ls"
    "Fae$cNyzcMZNdU{hGgk%Cb{^G{_Mh>=H`<el9t<-"
    "9VVNYgV?l?)Mn1XoNN#4#jq#`#(8`w?jzXy_QWHPXU4NItldcTcau2*^;%TBie<upL&hIZmxe*BA}YcNU`)rTiMhOqW${t5COzD^"
    "*_-QSBZoQyY7fZSF7f~Jobv<8m3pO44YDsFXKpt#C?*kEdv)>Pdo|6^`aUQiH5Amg5NJGsQPXCF7xKXUSg>za*VS{bc^1u2Dw&U-"
    "YiK+E{r##j;|at3P}rpLsjU2GD`J4lyl-xeceoH)%;Mo;H9=R0Iw^^r7lei!9_>%QZ-uPk0pAE6W)jDmyY%RJPAy7sTre88qs1D*"
    "USb~V^ppmiLRNam;`V7H&6j3$z~qP%^!k1Gk^XZ~uP_dU#FtnpuowKJc>Xz{0vXftK(_H`RLEzQJlxDQCz&3|h~;_jwZ%ETU^Lk&"
    "vUu^N?vU?JSWUzUjA+cwS=6-U`UKpEJl)jG-"
    "<Uj5UjlHq@6*q!7|9D9GNyE$pH|sp6^(NwT<9SfoDYP@QQaV%SDK9;TrozCgF7HTlQAnV8%)#q0PGb;9*j)kLWv2y5AisO+N?9M^"
    "mYq#KlEMkbY_o^l_dmBRhU9xH(v>fuC;3zX5m+HD#vy2{dT~$-*)b)a>M)}4Gn4MQ!KJ?;SyyXXkj4@>P1zYIUwp`^U2?rLualD-"
    "Y_ve!0Q%7!3&<|S+BA+g;Kq&Kp<zS(WcY>zs<$k{MkrZ$QEkc;DBHe%dbIy2877;)ty2TH0Mt}4LR{DVdf1^`-"
    "#^)cjpmokUB=^e&Ctt!D@H@8cUe1#Z*%aH}_v_LL8URdVD{VWw07_&NR$^XM(-PApt#`5$zno<1Y9?xZRT@&=DAYo-"
    "ucR=3%J2FnjU^R7D0ML{$Wa$`Ap5hRl}aw&6<T)v#%f4h-iY-yjk^HS+@o=r%*QwV!$O@)Q(!SC8@BQP?BZ+3-"
    "0$H6WQ;bk0x+n|@zdAIw3FFD>VaXbj=fv!H<^%06vDC?+YmHlY961yk999^;eM{s?a;R9yrXvkEp)<B2DVXXE9n=XjXbp1hW0B%+"
    "~ADF$T!PR@e_nx*~#B+KD>wjZHwjZn9jBNf83M!x;!;Bt(M?8V4uezOWT(l6R505ZY*q@_<lCI-"
    "%32B20>ldi)Hs=S5f@PiBrqIk2FjRhF0CMGzxViYvf%y6n0#fUui%%D<uHUAap9<|lJHiU?B76fWpB@MkWCJ0fi#c(`4oyRGMX3~"
    "@C9DcswLNwcILO4Cvn?!|7)MG@j2jxD$Z;r!dwrosr&n(6>^ClX1`nwMBz=|M#U!5$G_u?qLJTmd;U&=y<sW<Hop1H7pakfZVayk"
    "xTwB@OxoMEeGZ4r(*h}c|vW@~v^yn*K{?*8%II7G0?rXh8bYtw<Xj2WiYlsENpWf6lNa!YN4U5pn;G_k7ffuXV8qiB);-O&U{!-"
    "83rd_w#}>gACWDc+ncdK>!KdOn$-DD+^IKz+{dAy8Q9_i*kRWmTe;Wm5paa|8mB%G2cl$+6@zic7L6vvnyWazC-"
    "@P@y9gIe$i>*%xkjutChH#3uQ968cZ^j#L&g>hhVCriFn2m=QdWFqt`zpc&%p#|ZpS>+ip3AeOHBoj-DAsz?|cU~HuXBL_J~nC`E"
    "Eg<68#!OrJFkDdNZs#z4%AzQ#&6dMQRQ!|Ik!gI05fK5hr`Qm}y*xeQjoH+d$nn%)1>mjFbL*2Ii5<+J*G29z~EkaLa{tQKswL#g"
    "WwVzM|O`~PZ>Nzy6pFj|J(ryEA#`2&u_(g_uSC0Aelmj3OEHKKkxEEf5tOG19n$XzVSc3tX`{isNnDSuz+j_}>K}-"
    "`fBH5CcsJDlT+F`)n>Ve5P)grC8`)c~-"
    "HaGf7kJ}fdZiAUP&3hn6F*ljQ;)!_(je5HLAuz_W9jNwXC$4ciKz&0ow`*`SW+u$!e(rdeSMRdk@!&ui%{fS4;Q4s8&FpW%_{FYw"
    "n6DQFFo+KwWN7*)3uVX#CT`M&$Qex7m0**^bUi3W>AgB^BSkKpghY!JV{V?aqIBg(@6RdlJh?f?wGNMa#G3Q_X`G1p5RtBDSHww="
    "s+95!PE%R?*a40{mMu3&*-Q`==?vL_KJ>`=B2BBLz;1rTm}cr8JFaHhKdf?KRg-"
    "W>v58J#g5a3|veWw+KcpPc%_k*SdK(zGY9_l8v@V(c!z@L%T@s0T3fTLb&6Vina4Q@4r&FF|iV48b45j+Y<Af5d`#`18Nv{yv!5H"
    "W=FRZLodo!C9{*h<vRHw;SqWmUDZoEU#^=|v;mP$~%t1If~=`<wXt1(}yzm*`Nvy1<q560<M0ARBKgE{%#FJ|t}WJ9J^4pDRdj0&"
    "RaKzF!t>ivEMSxhl3SsEUS)+=AkwQcjUmm@uRE7p5)+9v*QcRe@{bfXgRf(8mg-0Trtp3G#ubm1^e`0Bi|avYEYOVgDTte<HQR-x"
    "WK(?_G*AsH6VPFm&lV3KVEW#;DO7&rT85|}C4dnIom`}BGMO;19ro8Y&=B8Ph>P6cRBnT$1Sa1eU+>A|n<Cw+T%uK5i1opfM@sty"
    "gZI~j-"
    "uCW@7A5&#YpL6YMIK!No<f%0&2Vt}4<KpIH!7wE)+YMTofV8M3l62za*W+a{19NYQ?!daRO0u!_LaS;C@@I*7znm3E|QJ@CS_*hf"
    "*KPD8|@l6Z3fh$U;27SZG1OR6iHzw7R8R?{Dnr^;YbVyD!FjC66WX8IG>!hC?&>s>;wgtF<0K3zA2XiJ{dHsE!CgsT=Y5D{5_nG+"
    "y_*TrM7(|Q!6?I=3kGyVvHi08c8Yt3Q5ZJTEOH26pEjty%)%?<d0<(w*Pj49KsV>>_1VX&{Jl83Z%Li=83@-q{n|TP-"
    "?rHZ#vxh=$NbidYR@0!zN_ZZ?f*Go`5Q-"
    "=YBn&<fU~WBjYOLwXjv*SaRC2&R%@YS0C7FWc$$Ugx9y7fsQ$SM`!EKJY9??~!_5r+OMp%m=I;@y(uu~WtmBjPS!n)=`wp)xu1og"
    "TLRDY;s!(HZ3L$LKB`{g~sb}<0%xguY(ur};BS4If^l6+?oG*^f=J`X9{gn*%+dr2khy2G3b3@a0@4jW@UXd!E+et}301N}quo2t"
    "x(lmhz&Op1aw@_CBrM0E%*Q4WvCFam~Jq$s7CqYFKd%8y;c$Spc<hCP3CBd%Q=VTaF5#4QanN7WM~=8t*sxs3?OnJC!uXik_~<5x"
    "05u=Mn_0Wly#8(`dITo;&DuGI?z*9fIMR6A|St^+IDo8*_F76qwb76wh~nzQ<zJ9<`4R1mcG_n0~VibwHc4wEOQGukX2f&ns_sQl"
    "Gv#N*NQV41V9>xfm*^9C~cJp7v}iD*YLmxz`lU05JjhOGDW<7~EyY@`%k`K2+o%SeNBM0SuJ8N3udu9vQC`66-"
    "9+RfJUT~Z%}GTR1*pgdJFx;iyK==&L$8?}K10KyPZ5N$Dx7oBS|#50>{XnbgQZaa3?3@}Ey@=&x+XSI3(wISu%3?}{@Zah*(Z0~R"
    "yDRVANG)bSHNAJE$r<n*y89*N~eb$z(K%q(hsZTMnWQ-"
    "?ye{U_PAwRNv2sJC`M=H3{)UnwJ#muS?q|dX=!bp>HR4Dto|KZFds|a&cn5GJ+v5g9Ec&$;<FB6S+j{KOUSI?%wnLb0r8TKOWU`D"
    "w(VJX7cpq_o1_A%1Nli$cBTQRoaLa=F@ee04!W@n=sWnKw`Whl0LiYdUj+<=MEhQNoXhRZ-"
    "SM!rZ7B_b_u{ycXJ+3Uuq{h#ea87!Nq-RSX55=ORAb}_=2!oPU4gbrpbrkS3yn`T1Yrs6zD!dBXFK<;1|6i+9vhoq-"
    ">z~axH!7SAA3qrvJiz<hoSSf1yU5q*n%`=4PkA@fE%x^BlH5U%Q*YfOTk!(9|aLB(@l|TXaz|&FAb`zB&!G$58I|4ssIgfQ9XWeD"
    "1L8*T|kCLg?S-J7wp+q6?Ep-zE*Q~FBxxIfonz_o|q%e5nv#6?!)4}?l^0gci@UH?Ii9S=d+;CZ@b&K(IVk^h~Nf^Me4ojw_Gxgf"
    "^>!*xv08CI+D45gnn&On<r}W>y90-"
    "hb%Sb$3(V3JZICDH$JLAlpzkn+$Q4e>Q(i)PFHU_F0aLnP#5b_ri0$9T}!+<f57ax`{XAmH@Fl!#{l1jf+scao)RWR!=hgiKCE}u"
    "yCldO=#oCxfv%!?a*5g_8_ImP3r#e}rwHBla45c_EvVRw@mK9XuP?{(p$MjNy4&T>}{5>(Zfm`OInc9`nbAc1A8JHVsv!ZxxQY)r"
    "rKH_VAO))1kk-MD`zN<dZY6WEhdUQ!O|^g;h6vCVm^Hs$O4+N&=c-"
    "JYuG>D6A9#SZNzV>{07Y8K$F+<m&`3BfdI{O#;sUV}yqc6USDUpUt^J3!kgot^U&5YaMep}J`y;-"
    "Tg5_7(bUcmhcW#0QfI0zdzrqBQ_V!4JaoF!qAE$sCtVzJ&UjS?%zsLK{(qLFVUbf)d{VUHc8Khk#@XF7W-"
    "nzkct;R|%)blf)XZSlnstb31eRFUoI4m*Y{@-_LbZdvLjv3*q8-"
    "WFhh*S#DKvWBYAK`}s#q);y$cdpt8!wTs@4b3TI#NMO&aOy!s)0{Lsq)X-"
    "T0Z5}_dvE0;blmsHo3rx#IY;0(8*>UvEBA;}6B#5$~2T=#Y7pAvxJ;u&23IhE{8?vXrCO>4aGO+ssvK7x{o64qrSs@S-"
    "7S;fN1~IERg<{5H*q$pGQcg495b@DA3F-"
    ")iBl1F)*GVgd7>#CW5g=%69d|qEQ<DUb!j()2A$)#;3(n8R092sx5es0!myjkFX7goVI_<EG-h<-"
    "#H{t?Pj#CzRZRM*4Vo1}hVT4hz4cL=*JDkHoXf5w2t#q4XAX&<{0PhY=7s1=BPA==Ddjt@(YO>2uVNN;jIzO##v3s5-"
    "K2a6F94n0tCTq$p@ckRHaiOWu2*9ViWCFeUGmOrc=3ZZK9S#5gX>r7Xq;N?PrB4nQ>BM0svSLLrvwiQj8T?en6wb7=1^8`d(R)T="
    "x(RYKlof33Q9eVS7N!VoOTnDIHmtGh#zAM!HSvTr%<E`E7n}?=nF<#@adHNcr(b(Fz@V%g958jp^8`=+nRpWqHf10};4A>(df;44"
    "LYJ&4I`{4pF!LdSg!m&SJxp2wiGr{nZD{~2IYKxQ@<o>q6CX!XHxR&-7fUMr1~7eu<8}GdB&X77r`QU$NTRK_Yq8S--xKj+dX5mp"
    "@(O}>6}8-=qy4OTpwru)#=^=A?50luV=!i`75kdz3teF62Q{<;UOS-"
    "a;51BU+=bf)yN%96cFtVJAZ8|Rhes8o8SDU0D<?^)n6gdJ@?|oguDk!^bPK|M4A83VE*sdPM`I|~+NUxsyg1`HhasK@LTESXZ3~!"
    "v%J!z2Y|W%128L9zMdjB`M|w3#8AB%CAh`uIifTYF6bLWoS^VT??gcO^Q0z(w*4zKHhYKU-"
    "v;T=t=Ju_Fz2_X%;RXvW(UBD{*gF8Ixmdr!7+z$@Gq-"
    "7`WobbJW!hilSN4t&88wKV?>C4Z0|95tEU!uzA*+M8ZL6}&L9TlQ9b?Z<T3NmzNs*cEF>8u2>kPWjB5=)`9=Omcv}r{)l)o&&_-"
    ")U5#O^;54mvSl4K5RJuaX#l%S*$2d$>=~%RK+rwp1%4BVCGTUl`1VS8-"
    "<T2n`TAaUDuKw+Sa0IWAJ+=yaLoYe}5V$WqyAD+CZ*H}q0h&3UQYx!7GN*34K14<`%iNXm7#t>e3A3Z$P7ZaVL)0%81-"
    ">)=j;O9h2S4Z{L-#09Ze#{hXc1};qdG}pC;oDc-u@#wE8vaC0|8{jPV)o4=1Q_hpm+qG2`<RG`x%x``rxB$FWOe;n~XxP=-"
    "GXl<A*zefxf%c%?I$GMqU7H7V+yi=+o*4Yt3eovkuk!%z9!EKjyHOGn$*B<-gIj<c^EVmggJZ+}(n-^uz3u_=e}-"
    "DR{PSQy2GOSO%RWHYGpYgR^C;XD9nCiA3Tn3{M#(gt@!5-"
    "zeroDmA&xLGl<ql+VWONc4;Ps#cxLvp$fL&ew6ro|C)**NqhXIyIy@k*{k<>b@0UF$TKf;935KuL@LnrCpibLABsF++LMM7@h<iC"
    "K^)u=NCw|q)fCV+B*@~W9P4i%8<}sp)Wf>jZ<J!ieKrn+GGw;R#0z^CTWU6Du83v+gB2O9^Ai$=Bi+@H-"
    "x*X=_I0DY}f;H$v(|Oqi1W8v7i3VSzNufnwAS#Hz$Bge$PIQk9=Rx)CR8M7s-"
    "eJJIf9KJMNf3uG>iK~y^+24imXjK^8NbL*%hAWctf^QKcX26f(TXW;F05eQ06@_$^Dx8_w;46+wJfmQ!LPzMnyjxuhGkB?ZO(p{V"
    "mKF3iFzhkdSP`VI#%nnq)w1iyTfFSXNuK=eT^0hf_?z3AGird=EhUM6x`Z1=n<Ds4yR;5nB8k^9hB)I`cL2<z~`?QlWPL^AUy;hj"
    ">d>>!dOD>{gk$J7p&~3$qjL=#_eJ5#H2AOa;x4hItWZY2rfmUCI|E+DvuH`SgxCkNe7DzEyiiFjNlMo`iU#t`kq~OuV-"
    "<TX#y!DC<_)rw<0SZm`q~7*4KPx4oCUWsd~)d?qvh8J>MpJs9%8Z%Z|XH&`=lBxCCCFg*dot<OEo#@}7SElg@qHMnbgvnvH3b%kF"
    "fLk)G(MgTDX-%%D5_8K-"
    "9z1j|likS<W_=4irgm=2Wd$(SO`WbPjR2jwYtVt5F__T;2V_#~ep_P@tuaBzcU=xL$ft}2}~iYIqHbL^F4f2Yj^@9zAp!a6D$1X|"
    "1!eMNF>uw*bMfOy$|_5o&Q(-y-"
    "`GA;@&9dU!?zU(t#iIRa&T7s=gG;6|<yHPg<b*Z^Gi2Jv0{!5t^rya~IhN}Y={w5%>;(M3{rVnnkvLRZnG^L*LZtk>UVuFvb=9qd"
    "x-yd+;-"
    ";9C{>mg8*nem&+Aj3Ay)7p>BzmQ`NXuVWQ(S3#J<tT9N3m?Et3@bE=NX6O9TAT~T8Ux3%QFE;+%xT!PQKuHoJCF^pT_tzLqNGu`Y"
    "#C|}c1l-?5EdB6l*h1)BM;*RvA{BQ^Wa8&Bc5F_gDz@=Dwm-gE;9`YpwWapY1ULth>!3K4vu{*{bH>krfm#hf9uI@cUhq)t{<h`#"
    "MI%D1JfC46>^M4#P;{lrcNdovSN~F`dsl%5GXtj0l45sR>kZgh%v0yn!NCE;<o;e(<@;N(}#fOdK*b?Br#xH(LHS|XBcBi$r(;2+"
    "kaUeoQiEmt|8hS{fu3@Az<QE(d@v}L9G8juI?;HmMlol^j_*Yi?Dg%R)~6OfNS6b!M)QzQB{j%6V0Ks^5%_jH#<iMkzZI>YZgaaZ"
    "jEO|7*9gwd0{2J+@#OgiX%PQB9K}3k7h#EykWJTS%wTV06HV18xGLx8AR$<)O>5C{h9_wY6SAB3F=^C3}q~1q1CoVX77o8u$0OvS"
    "M%jN{<I_(LV5Qwfq>*wLN}ch0=g_jn&0DiJi|&x7Xgs;%v624#dNO`kNW*=D?K+@d`DZShsN!N*@AZ4%i*<b>BNlvL2P$I{`6Nwc"
    "Kt0bK^IHMTSFbKYtyTz>tpAv<JQ87#d%jyKEp-"
    "2SHvvVg5x_$)B|_ItYpX^6$Er99o;I7C2y4Mu$)hE{GYaWa}4f;T$fg8QleGLRL^Wre>i*fKO?rG*>m-"
    "p)@;Uco5bwyIA0z)<3O&NA;FPnW_{_+>j(Y!udJ1v5+2;&Fw3iO6&k8Oay1A|@2?13-"
    "*2~J&7}4*104KIlJ4)2yt2dXlIjlM$Z<4pt+1XHx_;AvBqWdW&3(_$_i17bE0}OlKARq>yn~f_>l4by29NoBemN5EUZauAK2g7%+"
    "b|cNSakNc_w0pyJ<KSHGS><6F(F7K<!iLwGD7iN8HxY?={g6P2_<TwCHi%C)ARfElCs9CF}TkqExOAd%6S-`2|n0_l&-"
    "CwQ#Iu2Jrl^=5kwhB-?7QfRfKDmG`Upz!|HZdTHnWO))rf*<Eq!eGkYcw&ERwOXT72Q`2>M?=fLx8_g!K){e-u3{mB8+-"
    "NNPOE1T%s*b$Kp-rj5*sqGcwew>3nOW+J>>|r@x8oe#72v=%?Eh-e%`{D4&#O?UfJXly4s-+9sqW3Rg#?o-"
    "n@4xucB)RSJtqs@!<UoINMX53Z+|wLTqrpSdD5s7OCUyE<oWrnLgVNmnPr?*_yZ@sh^9|9dHv9tJ@OOB-QuYRv-vL;YKJ4o;b8Yz"
    "BRkY_hL5|W4xDMzguYQ-"
    "91zyRh+<H?f5VUgT6Ea^ocs=t^0A|=r&6|3In6&2FmVZsc=rKf;xLdatx2RAE8)IYbfeMtNA}K=5e&;+oWInpb$r<8nOADS&a+}?"
    "gn|fr&dVMkRR3RS8@Voj$tTC;T!K}eloxkt#c@io%K{jg^ZN499qlP5jNl@7Xy4@#N!i*YA2vl}NG%;Eqg;iLm6nT?XqRU%TBNZ<"
    "LZ~kw$_MdgGyb(TSt+S8PA;&|;L5MgdTBYE56MMO|hgq?-r{-Pxj(gH(-"
    "X5WZ>G|1*!(|^~lyERecH`)jBTd`=r>~^}dk+0@n*X@HcT5Gre3&ovcWAO(Ud(;`h<J0HTUR0*T5+#6|9v!OpR{BB#vAaY0}Lf<x"
    "&z?{3GwW4p0$A1^a`$ah!^}nR5>)*1iDR6B+^*2nA}SWwS;OCt1QiI)m8TzZh4r;!93*R-"
    "}81Kp+Agw$o`2V!ThQPNg*7uT8I45N7Wh*imB|99_A_@wGsDj@Uhm8BLa|Eu`Gu7VMYS}gz&%tMa^EWEl%R}x2(2hzzZ1_xbb&dv"
    "W*S>1ImF+?3}I388a?HFJ$rCcW3-"
    "FmFKk%b}^H$&`pK}oCkmfZ{Pj`!Xd%+v~LVTy`0@b{Yi`GE}vG}eA%H4_1bz;!#ZnKN7E?tlV1i2>yX}C*<DdE+E~?ZQxLKuc~WL"
    "(q_=<4qrhDE<bn=@UOb$wG+E5RpdIch!7!J4t;Z~r=lY3arhj=c-"
    "WeD^LXD0_IQ#Wzd_?cjL)IEX+v$hhKZ`DVQW6Oaoc6O_f~BQ<GWKS8sZx=#nE>=fCT#ZE87`x^E9-"
    "$FDDET>cCavpXVZ<aWwTav>vtlmH|4N(3U8vSYko`n7;yWm8JKF3C0d&xe41vuh6aWH{>P7A!@y=;<E?uJ8vi>+b9QW!MmxGC4>s"
    "+Nv#|?R@SImC_r@160$m1!%-"
    "!Px=rOh3z~=wfY*b@@$Js>;5WvaX1%RmwavDtBAY}8AvsJ*X9{t$Fy``;=x4d8K<jE0cJ_%Flp_73C@^D_5Wi9Q>8nN3_lZ|Wkwv"
    "JnmrfuCj1K=~5YK$0O*a_$HjEiI?z3DW%5Y@3U2;^D@9oO9w=EX7DJSo`|IcG;QKc!pZa!B48+qMH}kDoj_WgXt-"
    "*s!dLda6rrR+&RZ5FCEJ|7#$%Z!D7fs8X{U1fhBOp35rkS~uLM2EK|kXW>OsZDf?pz`4(K+Ds1Nf!DnkgRB0Ky6$N;VI;&!lzcm8"
    "J0R>`gnGE<HB1Q}1eGJm|Fh%&07C?uA2Hk3w@rJ72xNHpOjTOPNw&`xxkIsdp(YY^1%}ljeAk{-1!vlRepwA`x;_-"
    "{CZO{O$m2H2$#A|K)?hwRF{XN1?S0K1Jrr>A!?qBKk^EoJMi-&-"
    "c^zf=j7KkWv33MO8rt8jG)}lTGy9*N)Hd9pTTD|q8+A!|Gvo^I**g)9sO6V8bQ^t*&}=KhfRC15?w$!e%(D4cRaY@Z=NUdGtK9r5"
    "d*q=jEic2@GYot-"
    "{@)*=xZFGQAXht`#_tPm<uU_}x7n(q|EzZ5VFkF8IoKcVn;591tCZPdt)#^(Rh!*!6J*T>_Tu|J?Afi2G0zs$xh4s@pDIf~un9hq"
    "XUqc<m%u?e#9P&-8Pc(ELu|vK0EA`BF?hM}9>bWSBsO5J42SafU6EyGZ0)V>k;fS|-"
    "fIZZEggcSC8j0kGe*MD$G2EOO$U0giEqspX81LOYl2<v!G>nN)^U?s2$A*9^rP*uWlan&^!zF>`&}By-"
    "&aY=+6u9aPbHspa&6{@r)r9y{WaU3n9o%3cL+^St$xms-02Z9;zkB1B-"
    "!vJlk#Pdx=2+S8ODUwfH?~xmx3a5d+U}{2PRK#kMDRWGR%CjFo39)MLvD+lZaH6<2ce;LRocQwS4nu?c0soAHwTv;9FjV$3$Y8FU"
    "dRSne!hwMh~{1O{UZL&(Yz7FxM9Ix$G0vlz1Z<P=I%R8<R5=6q<dkqtZ%<!xCG@G8)ims`-"
    "?`n~v81^Y<7F6IM4pRw%fgu>~qKUa4VzZ?Iz%&2eiqngG}<A(p>u>FCwt9wZam=(SIfwwuwv?0H(g6yz;*b(P=onN)SI-iyrF@B7"
    ";XSUkePC6ur0R(s-P09eA#i9$rI#<iS$Ti5hvnSW7%EL(5vfLzlG@sczm5e(=_Nh*b-&LiV-%l2~3O!#|12hbxX-$Uv;V;-"
    "`hb`{RD(SOkvGD(`vV!aD_XXEzl4ZYKOyqVZ7cW;8!=9hd-"
    "<Ke4(<WKpExGbJp*fRdFRoPmj^x0}Mz?1Bb@lyOP5Oo&thT(oxkHm;5zUqk)dYjx{M<#;rK+6dHl-pE0lg17@z<IoLxGZhg!lF!W"
    "k;vj{78ylNOBs2ZC`gqHxChkv@UV4v7Kdcg?v1e9S8PY3&Z2xf2i`<IuQ&UZAf2J&w_}M-"
    "mi<1Nap5hKQ!%MGeDH~kAqngE_GL?<g?jDR;JNCd_UQt^Dx|-"
    "3KuaT|Kcb#>8MM|LlNfyN@qVp}fj8o(rNsQ0n^&Ey=vZp?Cj)q^m|<gw9=&9~^hRQZc+>Yx5D|u^%R6SN{SBUy(e0}&&9T|3HC{o"
    "!&8$~qgi6eV{v9~!`T0p8l}BfipvFY%E*-"
    "BL=5Um&@)^3LP;eTIe97s>CQaHaX%<Fy&n=J0ji9;3%6vd`E53_?cYdMp<@NrHkir32W>uWS)xN!{S+O+c;CHh%`5Y-"
    "ndt;hi&u^Lp8Ex|W-"
    "o7JP*=U?H+k1{L;<psJOiY75bYUL>ftqNQLdE~soR|V=2JF%Vejly89W%G49QvE+QJ*hdtY~S5AKAk73z`LE@VsPB6_=?iVr%-"
    "n19aZGo$f7$_w+5#$Dm}<^<0H>yY*8bsa1{xCzDFH&Tcw(<y2g5+vHDx__JV|xeSC@|NMKN{pw<~Mqp6tp^KSeFQhA7nhUC;T{Kv"
    "oHK&{uCwg`!7f+LMpBdt|l?^pOsVym(M5E;>Ag;1}O|<QO_NpN@is?6(@c6H(S>i!X8`gc01MJ8&%CC7og<RM(s>2Zn!7M|0Jm0i"
    "DHNLFoC?K>wp$oRrwSx)^gtROX^%EdL-"
    "hl66*HO2%Z%Q)}$jmj`De8Iy_ASor`i0DlfLyYdZittXpIRi+kX9QdMmvx|qP6z#FL|gqTRakd=dr-AaZCSh;U@V|SA~{mRWU(F_"
    "~*<A=E7Gy(h5$?bT71Xf)q3y9xhmiaxeIWQ#X(-?Bn-H^Fa4_CqlMMsYK~*0h1IzEc0@5L>jy(k`k_J%!469gz>byT}AI!;8$%-"
    "d5AAZOo>~^ZK4nm<($D<j$1S8W<z2dqkB-"
    "0#KL|PzN`8H;pO}d><CxmL)>lJTISpB%VYzNAXRpv?N?j+!<`^CHB{Z(_Fzp%WB)9rQFVlbCx<Gwb*vbHD>-"
    "Jek_~sOU3y{{)?=JC{5um&%kY}re(SJ<E_ID1I{O*F7h_Pl9%rZELbAoC-pMp*&^tPrC6t_PUmN@F;E56gDd&5=tb&c__qvH1O?U"
    "*__~vQp;04VAzPf_;?uWv$U^D3UzlL|3q4ckjfW_&>K&&P_*k|&0I<Rqz6nkZwRN0Q;a4EyADx)#ODUgn+P7bcy`qVcSB2in-wYr"
    "B_MEkvOzN!H$r7GQ&T}xXYvcm+H*vLzUKQ9qm3+TI;hMi)Ug+e_6iLiI=u@o^q7?{TQq@Q{L3gOjy<g+K;f<K)Xt}eGld^HtdP*H"
    "V&E#W_tf@!d?@ukA9tZS4O>8mPW!FQ&PPe;)!9_NTzhlZfMTfg5$F8p0ntUlp|Zp74Sv-"
    "c4le~<1GpeRjg9}YC5hJi*C&szh7NH)o5-"
    "e7s^!yr2)v@>>3$<X~u0t%=)eNf+cfJ1|Z8fM}5_|+As<KX4UFkx%krLgG@ev2Zb(I1EN+!;+M@~jQDC1c^yJfnDPPfWrt)`UJS("
    "-CipIJRv=QR%}L#4>0*4gJ+xZ+nBxrZ9q^NyOSh(mZI%e3G4KY|217>#SYJS)B<hDe~{OSjReVukt3??douPY1I7w@Y)eqmJH42="
    "d381rH{Q#n*;C`zS3`2rY9C;BwEc-a3wfhF*?`o@K+GaC#WD(hVJ2*t>L>M;b&#C3ct$c7T~1k)uheZnas%E2_0ZqQQW@qDR(v-"
    "UkG83Aw!Q)qD9e`lB$e>3?;}RDH)^=KExXFMjB`P{C_%_FsjK-iqkWq=nSv|RAr0lWPxO{Cvoa*3C`~h3hiwVsBl1OJL92ETzb#e"
    ")jSE7OxDYKHNE4p_c$1l*eB|5*}@s6-"
    "$sh7ro%MzY)KE8QY)+wqY3)<@Y!d*CX6Dxa!a;Gf*Y^s4PBn2*Oz>AXTApX+NLEv!x6}WebYH49nb=Sjy*l*sB!@q0<!-WK$-"
    "s8zoLa{3==7XWBPY7oXaOf5h!mNuB{oPIJ6``Rkj3I<6rhrzbYo`>lv7wV<5psuJ`K|+G9Yq(~z!h7Fh*QTH+Wg_n;t5&UlG!ItT"
    "rok$VWSS&~O`Vxbf))4zDxZhNu{%!}f0uZe4^B=6t7zP38V3+J#|)2~e^Q%u`yGJKE00=OU<vi<PQO4g4Z$;!6ft7;~(;k4~+0(b"
    "LG)ueRj@W_1^Jt-U!HoQsKhp@9HlB4qXoVZ=~Xue-HLMzzUz>e`9e+Tg0qt(r~9lq+EoRM=Si`hu8<1<*Mfl{O^e#xJqZKXS|RI7"
    "KOqMn{LavS*@gM2`4j7s~M7x@g#FFSSX!Di}O`ecscD5%fvHGmHs-FUf|B$>qht|_#YW{Kb{UBqKXj<|6xc%gK-"
    "<8V7iOY$bcFd~sLD<L*tv4V%`wh!?Ywlh>(%eXpk$bG`df8x>{q%O%%zQtX_RZAGx6!BEEA{e^HOJJ@EU$7~|#Wliw;uPkmym}9}"
    "N^rfmSb7XGW(?)urQI?B8eoy(@>!mXnj`50uz~k1h(62wuQ;Xck={PgVUB?{7kBLQeGzM*K$!2OgLmNxEE^lvdiHK1!z%Uo<yz_J"
    "g*}1sPa@#XZ{JNiT7$8r^JkBYdq@<1&-"
    "K|GYh1H8*>Wky>nC3mO!OK6_=fssmG?z|M#Q;a6DzZ1$VeF77;n^|=+NH)g0A*~+DmUt!8-ruuETqHwwb!S>+q`zs~$8_-"
    "L*Q#92DoMv#*Mu;)^r6{2OQA#Ol%|o_+>nt&JLnDZF)-"
    "yQRn#N_jM?gLV~9VmETPKwI8U9b}%4rHwCZ#i0;@EHlaR`H29yllI!KN9Dg*Jtv3e;2lVNf_Gxl-"
    ";OC~AhAf*%x=!9m&qe@OL0|A7SNd^bxZQ(q$ICMj0y<%S=UmETK_2;0xPlBDZ5;Vov|=yz=N2F0j5BnC+kgUAj_Qy==V#-"
    "41P^UbW1#+>=_9~Ru$3N%I78dXb3lN4=p{FR)^ATeM6Ht>TPh|I}7h`S@N3dCAnv00@}R<>_%$RSy9hj$pc2K`y8(Jub|`^PfX$)"
    "qpOlq=Mc_rErpEx{c{KP(7iF*X(9ac{p$EaK1uwjJrPcIg@_+SlnW653yWkF+G5an21Oha@IhDajC-"
    "d&WP7EcvlmJXQk}2iRf78PRQA=GL6(jLOJl2FU+|6uI-^iyg9c_e-"
    "t<x!sS`kXgc4<*a&^6Wr5G370mAXjWgY>%Cd<*&H(*#?E?=e%KJ7kAYe{U{O-$MkQ^@1&Yz2#YM{W#ImFbltz9xeQ^BX-AsdW{EN"
    "$Lx8Ihn@)^rdWilq#rM{3bF+u9}+n;|@`e#36dCMqB1;)b-"
    "w}!^hJ_!2e_i^PZI~SBlnT$*7{%`}zXw_>El1;vmm1Z%k6zsdMkOH-"
    "*yx*?$dLx5+Yhswh4aGvRw|X7(_UOLrnPnhJ3RrDq;|kC|T)YO?tz>As((JzJ2+phJ=U%!=%ZN?ge1aZNH)SxH>&TU^?usa)&D!8"
    "mkBw<bxLaFo$%sjEMeD>mx;P9l28%-"
    "A33U4<R%JNjN#(Y)TUn{}aY0};LUK+pHtML5ng0~#@r2^L!Ekn=D%eu$E3rx8umL%*E4UuZ75DmpbaN;U6?(b9OBGjJhi_Rp$2U4"
    "!M0yPLIQLV(^iNn&R8Zx27Hbwu<Se$b;f$B$Nn*p2IR3R8GGb3hSP`$#HmrIREbA$C8<OKRLZQSA+(d>w#DvakI~D{CFbVd25eui"
    "JP^A4rKe^x^CIZb8E1U?{t<z6<K~x+<6ZkBDedstHj$Yy-"
    ";l)SRJH4y5(>gV^Thw)PZv4f@*L0@Nh5vhM;^d<fOw&))}vzXSE~S4mN3jy(%nM!1Ld$5^Q(*ATG;&D&3bh^Oont^~!t9RLSGX?x"
    "5DzPD3?a75uuS5%{H8!~erF-@XTN6sX;$hPf{jm5u+z!2mloy|xsU7gMuv-oAyRgo|*!3?wl9K;JjQj>c0Vlpv6OB@cYV4YXEC6o"
    "E3bf1y4c{LOKmBP8g63vMgLt_AzWtL4@U8Kjm-"
    "d@Z<8J{^UFVS0PQ$G`P@^~#81shJkr!%(ewI#;Fi0JaOpBZ3Il49wI5nfG9^7pVcz6o6Z9Wld{ZvtEJ_xp54hPwKDz^M1zbZK{fA"
    "7!qwv%9U=MH`6usY&QLS#}ug++Ndr6oP@@27Kp-"
    "?LL_3*dT|9d~I_`vf6RpExnq{CLSg79d_m3A%#WEqo5)`pU)<0a0%P&|J_Cjbwfe}-"
    "sS$nYHW3by>RutU5f9v)c`qK!Ct6o%dQHZduxTS*zgZg#CW><Sod<6)0~O=?f2GnFLKuN?wx?G{?4_;1#Am3eT`i6;|g5uEvo}Uw"
    "R~*5i*D59`V!(H%P(!T4~7ejfN$chnF*NDQG!=<>EYtzH&zOnweBEP?-"
    "FqEdyN6&@zBf5Gm05m&XurlRFu#*`@3b&N4H4=^kfsqKpQ!rf%+J%+3wAn4qLcmk`HmrIKUF-"
    ")R8eLGFLFX%ZFT!Qzyd}#pOBxb(2kPwnugddxr@`*0p!a)_*?7<FDv;77t|Jj6DmiCyTY3_{3EYG5RZR`hHwP!3UfE{i?dpY9*1x"
    "GXQ@-L!5b^VG4V<YI$Z5dw%G%k~V4_he;Ah7;kUA{cGq&F-5zmx;m&-bp}}-"
    ")H7uaYOisF<l3VX45@smg;vbhz4JNJX&V`W4{Q}NQ#(ZZE<MH2YBdLX_BEP`HnKlY%pQKzYn?F1tMsH#5|Or%yff$cm~_b7GJ+~w"
    "H+o#B=DFAB`Q$%9*%;f=m`aIaq%S^CbA)(AjkTQQ`&i}ZldPshZJBS*BGnzW-"
    ">H)wS6^bt8W{MVrCm!a3DWmOcQ04#<qAC&?`>%%0I2PMS4x&Yz4Y%NJ~{4Bm41r}f0vJ2Acxs|bQ=Ah)cH$ccZ}6UftNN2jIT{+4"
    "0a&u#`*goZejqDi?E$F2B>w1+#e84s;X@;XZB92xxJ-Oa5W)&J+Y{F>XC3+D*9}eyWLUlo;YR}?F~P4GpL;sg7*<qiA48uee?x%x"
    "$T$0zlh8SySJwXRLHS`7LPPLj4hF#T_zoYDK8P8e~fytJ~Kix;Csj&jB6yR3`@VWZym0cu@IPD*cv5+f%d%OxVTDGT&au(JSHbgK"
    "hlvOFi0CN9do>)sv7B)@%EX2Xl=5F$on2&RMhK#Cp7+P)guuujW3z2ZSWkKL9N^D(VaIK5bxcOPegOg--GTbWJsU9Hl=8<e3O!JR"
    "f6;ZDDN5H_b)GXi?mj0PitJqU6!6GR+_0<;v)&4kv?y2h$4@maHa%?#_N(%W|w^X>{R9j*QyJVIEA&6Jxz7G3Mu4lne6VD=H!Sm<"
    "K0K{xBDP$RJ5#r5IVri_pGSWchuoBS(dC%-7T%Bdn({^%>;33lRPbcpmWJKu%^wfD8s;uh_t^T)3=9vUx(gr-fkw5PiU{-"
    "zEmp`8m>wNKL>mel+Evd24%c;7-"
    "rz!T|p20d=2eT5of5kwQz+vT#zu5RToRyr=a1*f4EM54zgq@qnjeC@8jPaL@wlgH4B*>$goKW!l33GZL*XCn%Q+Mpi8EX`J`BE&W"
    "}XvtaaB(BcFs8Q-SIQSvv-C;`3oL3vzCn4e~Jm99e0t7c-"
    "_Ftv)8vT@qs>MFWng{qF|U)4=ePdUCE7f1lY~5dA8PN7a8MKakY>qV;R(O4-"
    "<=o?05;k8?+eV=%JK)%WWdIpmWR^fteS2!1JKhBtnwKRa{(RV0n3Pu-"
    "R5$Z>Lf;BN?8Juzn8j2fKKfJf&U?p$90H{?hlbk}rfb|avf;G7<Zo?ePf7H=lT<{SViLuy!#LSuUMkU%{R=G`*WC040&NEH;f+)Q"
    "F&*yv25rn;~pk&H-%eSMkZPk8AjF>902&k#xq0IwYHE@DeOS<y1OWKW*1A4uEUGk(Ulpa?NOhBlO;a`3fR)1w_nDP?)oW-pr!T0C"
    "3h76xWuyiP={D|Z$nD?w^31iwDCFV~=M0bgC~J@I!*J%KiGUGx^<iL%uQ!+*D(&hY9cv_znN651UB=24|RgJKi@awcwug|IOW(Kh"
    "c^gy%Kkac}Z`_awia9r<;IEZ|W)y7XWPBs)f`z9YKsa$SF8r^{F0$wJy;11g`zHXp@xCIm%sr<kpR)llxq0d#jH;hM;abZyWhGh@"
    "G_1XJB}>Qs-j1&Zrxg}<ID)tcnQVDzAN+Vv}CjS-V_sb3p|T2seN|7Q_*ijClZrW)_b5pKt-<<dw#g-VW|p<G!L!{-"
    "hPmRL)C_Qj*-plHlw^Y|W@Ez1+b;mXSCp}{2>4;?Uu`0lC|a8UuD8HaYUY+HN<?|F1$s&&7@z_h~gKgJM=_X@t>-"
    "_;v>o|_=Rlr?)xv~+rILfV;_q??KtlW=8^Eqil4X{s{EwV7SVj=lw9h;V-Ft-#oemncf`X~u>ZDAet&kLnGj5lmwIE>o%(gI=$|-"
    "gsfVFs0y;q;OixPD%#tk2?8Duu@}|JX;v(E)8YkHaEF5Qp)$0fchHaogq+JgHSuqB}csHu=YW<?=JPh%TU}cNt=Rsd!C34-"
    "^tU>=!$h%i!|X9REY_ACt*qIcrSH>0nLu$nG<}E!|c(fyq~`%3~nNu?c?{!5zGuT6q-"
    "+gy$1|<5*6G7a6k8mPW(LwsX$d1_GxJb<+na#XY--"
    "x<%apKhkhbv({MhmRi&wdiQWYcjFkzykvh~S{OqWgL*%3;+Olzcl?2&aV$xXN`vf;MeRUM3FZC?!R!OK==(=x7SG>Tw<s*)WbNY-"
    "hKF-RbYrfT<fS{7t3?a?Nmn|fahP|)X--o4(@#|sR)?w(br4VSFdtZgri+)Bjl(pSE!Orr46<c)<`%ESAcRn;EZ^A{s0wG+%4q0$"
    "uzCpD_IX276jbek@=F_c_m>WdDsRwIeNMq|3#gjOv0nJ)hFQ359v(t%PHF?eU55Q|Dxh0bd-"
    "l^iJ&LwPPpw4}S81w`Qw3XUeeX1EZi|L!q|1-"
    "`V5_s>9O>n3P6?<<mf)9eG`%F;t8IKg)QI#56PAlN}!VM%FloV;inR}qeqS21TLvlMs%Zr&@CVeUS*V%8Am}6$oE1^40r`M$<Ih3"
    "{NU3xOUY&9@&6HhXj!Laa^*NpQUh#kT8Z51QiL0)MomaQSh<-@yE(LO=nECb`r!r7=jMNqS8VV|CmUyrrtug;z=Y-hZ=@`2>)a}-"
    "`dcOT;l;NP~Wl~}hlxLca8LB;Qpt;>VI*Sd~sjCW^%Am&>E3~%kwhi<tOVZ8I%bZr2*M#ta>8*t>2lwtqk9WhStf%#Hc6sS@APqU"
    "mP2kN?)_DE7-JdvD5xBbjg%T+8a!i@TyLrvP00pvYYQ%np>k;^Hpu5Ky8?-"
    "^YMjAB48+h;(55#X&xm6*{_2sxU~=rW$60lsOyUrXL<pDo=b#2HY-c~1TWS|AJlmEe>rVjgh_-o^Fvi&K-"
    "o;0ZoW=E3@u*6fCEezY!@(Hr@DGpOlem)Pft+uCCPlOZwvGS6J^*4_h>7G;;bxu#`g7hxvpls?W@Rwkq@3}fDqkFm<aEi6ZMKJND"
    "=OmVWE<DQwUp%I<bXz2PqNe+q5`slQkl|h1TQ6VWeWU`<Lxq@V}RBwl|)Qq<REu;&y6?sj>>XKp7I@RPRf9pu}()-"
    "M+O4tGFTY`^=WkE8n0P~*r0@zB4?pm$h-kT6o-"
    "#5kmKD0#RvsF|)Y0lZZe>zi4hnL^7VXl}b4_+V7MQ4jhKnWr$ps?}|?D^C*wZZ`Q89Lh+l6hJ3_ucq}1EhLcle=EMyq`?5#KV0Jx"
    "^54f0fu{PkT~wh?>mGQJl(r}Lxs!^VY4P<q{NIAh6dpDqXmm2)_5XKY~w-uI!wq^|4&AWWm@(#J4ViHR_=cP-"
    "`k_{weGGX@XAJ6iVrpNxO<dsIBqqbO@m?_0;FkNmgxtjN*3sQ<Uz|=3@ivlk5E26v%v}Q!X~=0y@@}!GgC;}Rn`$5jcFXG*Vzy0t"
    ";|x?V?@+r;$9+UG;zC+zZqjxFUX_IF$a7^3U5+4uMUSXj9n8z8cSY_iU^b>M>;>wWd_<e6OZ}%ZYq9!j3vK`8`$<>hV*8?l$~45*"
    "q<?*^6DsvT3`MeX8%OjQLR;wK}Vq!#0j)Iou|%bK0Kb$ry$Oo?Ob0fsB>h=a^uLAS#~VJ1KjdP&hLy@fv!-XOmUKfoSnm(^gp-"
    "f)<TvntErY*YL3o)NmnG%XNiQm+s8~|kr^c0O#B*iZo<TPO`e$}y~Zu?9uMGq4*VHX_6Q?o|A5l|j*|%ewz#XG=oQt3N=hf$yX}m"
    "dNdL6S@0S!nGLdWa%ru|}h{of|oc6PB4fFAr@W3rWptO5xNs)1tJEc6>=7X;kHJaCwZ6^z0<M=Sc>L1DJ0+htP#|PgKd%hU#^}Ll"
    "e)^g%&yw{Q&H=pbbL*RKOXbhPs^`q6Mk9iwfreomN3{dyi9m5JJ&y_yz>JwS#YD$r)?tLc4dg(Nyn&NkR|C-"
    ";hO6Xs^O$Fx_FSQ|`?UQTf+|10!)aq$8CMKUc%3D(%j6)1huZHO7-"
    "OD%%HdJ53rIu6anYF5KO7dD<Z?5qD|5sL|9Zn9np~=2|GtMXdptyEKvkne1ZNJ>SM!!=WA|!IW{7WemJYVpF$pdYPxs;f9i6KdIR"
    "0WJwNwpN^()JBeXqMmP!Y+49)Q=D^^fppWpI*;rTuS2hp4S(So5TTvpJ;Y^WSV!<?Pm-"
    "7wj#+ee+vxrYEuzj2`!xA==Ozl^UVs=fvPLC0R=vu?h*j~G*$q2RHWRcTb-IgdgOuu#O@S9PoXjK`vfXg-Dr}cxXMf?{@khrMls?"
    "ETdB4++zOC;IE;2+pU!i|lxFk%{_ul0ZZbCm;3WuhTAAIea!-"
    "(%K>|Lg>aa<<VR_&*WUJYNzB4O#@7mc3cPJ={W~1yhq3rXCvd8SjkpK85bD2h;B##57-"
    "EEc=)w9^GPTxSIP15+1X;#^1PM9#_C9e3lJfo5@Oy6%B$){tJCF34hDfTRV)irogI5g4kn{r@C!kP@&-zI(*A4Dpj-"
    "LW@eEdiz;DVKEli()e2oJ8SxKXbgy;Xn^DYa6S%B;CvMOpE%ob62$E5xD9*g5ji;n0pnxxb4N--6LTnJ&_>#rqJRXH)H==X2kg;E"
    "~Y9W+_alEL~vL4;AdhK?nuS25q`O0o}e<^Yg_NFY~xRND2GPp@U4t&W|xm|wk<?ga`2E90zK)Y<@Y!(fx==V)?g`ius2h+wbwzeR"
    "imX?JD%BlF@5|_eX=)kEL{aEh{A&<6zl58o?I|NB1W%k*!@>}u=p@4FGM3rRGh4U-"
    "|P9#(Li`}LHBGe?T}#Z&uJyL+3mT&WXQ=XBosU59+q)R?2CUpn*}U;bu7KuJB}gq`-"
    "J@gvRqy5qOwvlTL2vsUc_eQL6Bm#n~Jm!Pc*kF4FOvFr<{V=0#w)UJ&&79_(+p`Z4)@0L5T>?()EXme}@3-"
    "Y8W7>Z#M}HLfAgtnK6E-#Ye-J<%_wUY3}!>Ype+C(rGcqHd%}0c2{hl;~BL%N8&zeLEDkY7t1MARlLV{Ybnk4;XlP^-"
    "eW2}GS7MgmU~ArXFm{vu`sY>a6ZWBLQMCZ!1YE2jcrqSZ6Sor^Gb2-"
    "C`<%fYkM8l)WRwQRNy}ST`g>gIm1VU%mtdTTgCM3EHvaz#_H(B--"
    "Epeo|HN8fE3R+^_Z8}$Jd2$2%@$lW?A>w0dj8necoDISKE)~^QwdIXSBUAv|DOdX2!;&_#H`Ety>!foPnu@*&cPuJPEeRl-bmZ^="
    "B)pqUN6W@V}O@;aLBbuO%NdH8ic$RyPOMq}V1HapUR%&-"
    "&K*JHbNLs@d`Q+)h^HrS}k*UXI{e%GNquJj{=T8B_o(Ops0{+(^r&!8ZRsqb<>yzrW}mU6Mei)jsg#IL<)sP8%tVTp2Nw3w1B{w;"
    "Ho9m?OwPE9W;gNgvZPm9{KwC)|idS0iSn+r~IuT>#ui0+Ty$Cx+j^RyY8zK{mL~?~olDH|;>MwN`MN2zM7`ZH4)Ke4#)%=_nFQVz"
    "jUM`mE9Sz^2!aPACaUJb^}Favs#DOgVA0WYt^aqATiWph7a9#)0cZ51U-K+boLrz&Fg6p86N-"
    "f)1%4l6;c9Og3d%Q={DjL3to%j%3P5v|a7`5lXMfk9WFZDOSm|!YOQCXZj~;>KFi>Z5A%y?<aZ^1l90S5Z@)z%HMyc1Iv!QNK^pJ"
    "{Dw&%cyhssw)vX@vF50xQ|R<S*hzu+VdFTFJQQ0z;0N&M{UPo7N)5Z51@F++2!GS;IM*gGZ4LQ;tzcSn#U6q=oiu=(DC^d|#VcS$"
    ";2^AS&sP|PomO$}xh3r+sq+@DSF2OE+Pmv}t|a7iO(QGnN={^m-;;Etk-"
    "5_s&pJpsML)YGk8jyWBWhbN6Tu^RG}lZhJtiMi{CM|6KB1lu_@}nCQFR;s5Lg%fh&jNg6gNj5>M057OY?s&U&bU7QZVcWNN+J2k{"
    "Y+|sol@P31onqj>B@aAEk|rD61##hmlX_WlxS`u;o3Lwj0zw@TSeAF-wIf`Nq1R!R*)&TbS=S8WZ=O{0A9RdV{-"
    "(y~f^_GH%paxp{{VwWn{cRb3=*1$g;L%7B9f8D^5Io9ih#$8K4)e$|F2nRw5O)h;e$S{7xJ>Z75_M4P5_ad8eF2)8$?N0Qw^8SaF"
    ";;Q-+8b&Y5+=9usZ%%I1X)M(rDb#=bE@w)GU?PqHQZ?qN$8;1k#qe4(MIliMA-"
    "M?1v_j10G*R~z*N(O3nvnC$vglRZuP;;)m4{=8_h(XpUWCVRgD&DCDj+_<8TnvXHhRs9Qa^|PxNkX*V5}wG2mL#(WwbXxh4Q>li#"
    "?RQfsfbh1AC{q4;3}n(cM2P65hG2?O0r2(PTuW$SC*1wEV=O4W~)SyXMY%%*$_<4vX4aS?(g&I${!}I>yOviAaIfrSJmE++PW6CV"
    "voqIVM*vDo9qgy!$z=m?$9FSV+@eDPYC_`o=}Zs9PHiFt_Wj9iF@CPxZY2W?Q6SRbIAmdR5Y*hHQv5RHu{V26hI$D5Emh(R1J}!r"
    "w|W0D|%3}gTf>>(<ZGxG$A?L($J&Mli0$P3(BJbIexw~4_=fwBYX|;OaVy@wzjNq%?cGHp)8h!HcV*P>mYJNdfQ`_2t=f<m}j;vO"
    "3G&&vx=Hl19Hi!6Zu&CCAb2=%&GmI9tumq=Ev*Y0LJ7m*zbYow;$0gllaH2vJGj7sCi3n*qPg`vlNqcBckeVXjw1?avrV(?0rG8f"
    "p;pY&kQaLD{~NsFCodlJ>h)AHkRziuzF1b<pG_jQ(fPWTl|0#+6U2~Y-"
    "4&h9tpNg+0YQ6OwO}ukhxIzaJnQ+`zP_w?MlG0T#tZ8&yK1$K}#c|{8+9D7PfiLTtYHgBgmcCHrjWVwmpuwWM(C_66$C@cf2~47-"
    "~D7iQQF?xGxDvIwZ?Bg^a8F^A_1TSd4<;_TLGfu)DY-Io@h6C7h0;;!2ra3h_IJ@>>jooXz3hsj%)yekOZx&ktkz_FR)#r@;buEe"
    "KqjK;6ZGLqKl6J-"
    "oofcCOtSPhArb7nu+>6ZppTHS$EDYr^1CvebX|C;{`Gko9YZ(pD#AXf~;HWxXNVn%!)9vARxq$=KA&^mRQ`dC7PPcWdF;O-"
    "oQoOTH-SP=jpN%x-"
    "67AtmBZ(83DcaKxS4RE4|XD2J}N9_%2$uMjlXjvv$>aC+ob>wg9$DgpaiL(w1C{E`S?6BJ2kTepcvRvC=$1v0Yqx8Fpc1W#?>Q&Q"
    "^@_gBc{Iy=Qq<r};9aV#ft%uq+UmK}VD2WWUGN+BHg&$e+H=Rw8I*7ewnUB*%c12)q;VNzE#4A-"
    "o%)uui|G2>^WUT8gi4YV~x#?2Xh6=3c*frnSKOy0oOKnUs)=I}|rfi2fsKMtJMl27UJ@^>o))8;CH>K%33VUT`i^!_XI?l!J2k-"
    "|!gczQd)4@1>{1a#K_CVSLb!9SNv;2Ae8OH4keEmWpnr#}gtw(IFU18qWHH8CD%Ye$y~O?mf{O%BL%C(`D71^#LR`V+wtgD}O$WF"
    "tXc5(aa%p9A$ZFS6S9XJ>!Ci(GlgdLN^sk-"
    "|qz<tSffe8El>tLKtU`#JDo<jCKNWzsWDVRaZqye9oxm(i)?gi=UqiCYS@Mhu_v);3GGu#4aLN;qq38I}k#yN1AlzhzjKG?`AlJu"
    "1}T{7U*eikqRcc-YO9l{!mTznLIluPS`D@h@yab!;iXw=S`xjd$b(bv^NPM?PM*vAq>!=be}x)D@H4p#&Snc?Eq?>%?`V##<ioq0"
    "FyCF@IK&Y=v5`70<<VB!uocCT&7C1E~hw=jYfWJ@sgW@?Yk+CseE}{oyf-"
    "Tkh1ThwgJy*M(5QU_@`fbSwsKY(|2XXsv1uWX5}Ekou*GGR$m&F^nQf&;DxDna-xLJou-"
    "AXLZu_+9ofg^r$EP%x^i6N&1O3a1KHmg-T)eTmjT}E8(;}Fn>i3tY<@iY-A-zN*t<@6Gk_2`jrfj{&g&oG4YSU+SWF=k}Kbwx>v0"
    "~L?KerN_C!r0eI$hr6P#g=ASNdc_=9TG3sjpw@Ef4UXJ*A!q9>34^joEXm^-T1xo-^{q^A?M3b%>JorQPx!nPR-"
    "3A~*lV_s{#{K(&Y`)I=4S5;v<WIPUXJ19Ql*}rG@4ZJ+wo9fu+WIy?Ua9&*yASp>hcxhQR&?fS{moJ-kX4|FUL9PvBmqScvqu5mm"
    "a1IHKA`Yg>Y)<%i^r%L8Bh6cJ0oH3`YS+!eo?{lWAv(7Me_T)R}rH|ijz4ILFf}N-N*aRe8sVmaPR7OkJ>_MP42OeU&+RD-FfwU@"
    "a-B(0?L{7La#D45u9-aqrE<R9R#hXHE^1Sq$b^Sw|FJa>F_IbgdA@OZFf_&w%DtUS=KhC2ozuyyApoTi`3*g5p^WNxy8MZoE-"
    "fw8Q*W4=X0JOF#wJHD7i|y828k$0Inr-aN_^pCo-kv&Alj}CuQfzaF%kT=>Oj4EeN$(S{d2;<;z@$$sSO*SK7bhMr2Nzt*X@ujN%"
    "U6__-%|Ed&%tUIn_{)3%gOL$uE}2k85x96B?8yw16C-|zu&!IC^Uu~gB0`-"
    "Vp+TI)Om328R1z>EUkZ08J9syAFvi3vB*_B`{*?8X2fEJmxfX7t@AoZ;S^3hB!72%_9sxA*ySYHHv34BJP9`pwQ85b(Z?SI}9hs)"
    "=lDP{JhJ%8Day>5{R($%(*7a_|hNX-$Tiom3|c8O<*E{c<<1L7bWFP_9~aUQL=ymgO63QWKHFIq{c><-"
    "I16n`t^5?d^|$FGYRJvXCOCcl!n#bWd+C;y<WKbSt6`PX4nNk*mi-eIHBj0x##`eL}10e<tH7P*RKBrx)>_&Qu-"
    "m^XIXDaUtG}T)%cSYy5mZy$kUlLC~_+u$?L)s_evv+x&>jezIh<CM*?i3o1zR$Zg{+Z=cCSflXq;W7aE#-"
    "(yCb<*(g8H9f61%^NX(Zj<g0GJ4kl!h7$7BrTHPeUA--63%T74|&d9Q2+`2SF%vv$sxcY36)S@#H{D*^~iEF@%7WjyC}^KrC48ez"
    "mXrpaykfvNcN4erknU3H!s2S(GwP<PU$`<t3CU5MjAr;janxUC{*orte0I<zcDfJBP-lKd<ch(Bf@9UeIl{?mMRy>?HqcDzz54cN"
    "yF!u%0;~HAj+{?^!bnv)Pwl1HI|;lVlBg*_i5$z`b6-JLo^%TyI%6Zm7k?U2b3NOk_g@I?L-"
    "Y@^kn6zjz@ErqUo^(osWa#ZHa32g#Z|zVtC5<!DRTIVd7f6?H`}XiX<RG#8d~~y`9L=Oo!yWZhycT8?kPe19HR{e4Jh~C;zlD;e="
    "@BmkPb=?y=K%_q1&NPs2)kqVn+THl5H9v?Z|y^<aQ8RDM?iIxdRx7Ob5iK;Um2{T#w?-"
    "a*lFNbuec9sv_dU`aIcHGf%ZU`Tv+^`J(*cx0=e1jBoFD2%dKU|4)bBU;~*q>fH>KSQTc5MnxHWju0nS_Iv>EZ_QqiAN$kHk$Y@F"
    "A715+QV(UwRPT9&l-"
    "|ZVQ+eAg`NwpvC0^`ZVU<LMyyyZA+j&b;*_Rh$X*Ofdvf67b*yGlFR=x9{!Cb=nZAjLHg%%0?U%@wkmx3FBz>!1i3w-"
    "|y{`n=GcW3ImPz}?t$xlR&Sp7BFeyLfAwU`kEkDeX4bykNT3SbFq9dUS8zS+YCFfNk@-?@jg|>;WOKj#1{;exPMa-"
    "C0tRqPq4)LN^KP+{3O9Yb1ZYQzV0QuqoVxoU1emMD+nqtR%`&<^YLWAoidD8u3KH=6Q{@;`H;<11^V;5h};Wd%EvF6CInazzb4tq"
    "6}3e8V1;x;BI<HIKe<uZ*%^tEGEv1GiH2?IP2oM-CsT)n=hRZVI{e$n<ZWr25PUcZ$uA+5AVF7Wd4f}eMo6xr!N*MLlKSDza5NfK"
    "h+=+Wbx@gj_4F?^>)+xtek#Xs`ZyY$VF2cSt`#x{`#t&>WLi9rI>vOeQ(ifIQ;z}l?AYb<8eoa(>5X2gW0TbUhH8}(l-"
    "v$)$rdU+QZcZ~<)RW~>6zvdz*mJa8GZpBHG6f0tzNq&lEAX+;^L;QW-bW3+D=+(niW8Tnm=(>iuqeQ?gmS?yW90A5mA@sWy*7pw%"
    "m+3X-"
    "qcqyVae{D~$gD|V?SJZfcrxrackIEuNr&OOrAtrn6WBnkMTLQEI4AMWO^x45Y1w*z{s<{$#$vOMw&*iK@>881*A?F4lL*Kl4C`?t"
    "nQ(SliIKoZeCEG+#*_)b)sqbQ^V)t4hizt8`V1r2f!tKVj;t%zai=vrzaKs%4@QSxAEhkL!-"
    "lx`4h6!)N_&42+_xQeNg{%f@^cnV5WA~?{3=b`IL9vRrsXhQrvpKeHkh#C!yl-65y|5S-"
    "?0k9I$*VU$rnx&P$TQrD1%QPJ)qj)0_<4b9Qy>p8=_+zZ>+Y*S67lcIJ(XcbClan!J|Rw??#cDi1yzzeVntJ4uxP#n)x{ae<(>@l"
    "ZV~eb<A%0_w$&oUo|Ff#x3Zeq<aR128=;;1zbm=-"
    "^=z)%W<qZz5ejCieh^)^P#?{vcS^iV_DZe_An*i$f6#HLeG&oU&#FLYwz#`@L*iXPP(gE`R|-ExMbJ66M)c5Nw5?;-"
    "NMAyV|6MLX-"
    "1&tn&Hlg#&e<(p;YMR{Yu!7XfCYHVVOO|u1BP$Lk&$thv6%8UpvUC0!f09m#PeSMeKUz#7Mnub|vSzv)l3iD4vn=P1f>OQ+IzD7K"
    "#X4$qC#mBsk+%(x@%zyF+CR_~5kAAlDii5>5%Rq*HL8*&b=ttG?LZ@_|cTy+UL>gF$S~{*kmgJ6hAOYM+j5BTN&1r)sZ@au>xvx6"
    "u$!!L8-LQl5{aj8l*;1D53Kb`{t<jGe?5k98w%{_QD%;nv0M8BZgV-"
    "`f*o$Lo`LwPX8}ey0WvD&326^v>hS3Q~1qJ215@ugmWUk||saC;j}fzg7D|u9SR)j)ci+moYMH^~AxPF{Cl*O$1}op6~S0*I0OtC"
    ">!WnHF{sXrQ;!BDK+w6U?+*t688PX`a16zSUCsJcI|CRs$U|Jx(6AUp}%h}j;j4^QKeok2vS(E(I0YBTNqzORch(yj<Etf_$-"
    "}4aAAULRqQT~VIn)|BIi#x-QZivNd`9Ta>wXF_sC*H3%X`1Ak@$54L|qGi=&B^Ubn9?c-M?$3xG~CSa6t~z{FW7u@S|yqo-"
    "o*3gSG;WF5HW3=2z*E2_m5#7<BdVkp1AJGm~UVKNq+1Z!2+o*~U7blUDk5EK9N2$?eIw>`+DrH>OT$2+T-6PvC-VbTr$7Q&nkq0`"
    "Clb*9lH)#%Nr$uh#$@J~xEMZy@PtErnd;E^Kwe6C)XqB6i%)E;N#N=ZE?f4(O8J(WwWA5qSCXwMy4n8+?&eAmG)RfO5ok5FH2v>!"
    "**({xMQq)G~(_EIUe>ONuUWx8IpC4Dg~ud-V5*Ehvl@x+p-"
    "GGZ@i14Ggx4X@>8^voL!&$2$gNLJk<UO$1|xWc{wRLrI1X8ACB4N267>AjqlKR@QQGioXn6>6Kcd0tpKBu-4-"
    "LyU)tFvrRIum1d+be4$W^ydR5GN_6VP3-(O=_^-yzQE3rF}?y-pxtImdG&WBS<9KmI=z0L(k-"
    "6pKw=sF?{P}sS=O0Z?L)`+T_9|;*2x;`Y66DE%*Z8L4H&`I8w?$7M!zm}2%(yviDe#h@b4)SK>xl`;j}aGG;s%Fo0KY|`XlI75Xs"
    "}C?K>pp*ZOtJ2mg|s1Smn{a*ZF>|8GFJ<^tRIc(2LNGlRY-4-_SWD_S{PheTp8*2IG7;d-"
    "EQiLkS^2!*l((6R`Y0=z6=(Bv5&nf3zDyYCxGXBCPuNMiSGSo37>L&$M~rRR$7VKQnD(}61_$hGPsO{6c7(r3ImPPLtwDJ|~Ru!e"
    "#i-Z3Hd`oSPbsd^_bvPy8x9jP;o%Q-AFlOd;`73`QBTxFPEqP?CmAD!Zx+<@Wr-hII*0?uH_2hp$emBlk*{_?iMM6V}Tl@vds&Yt"
    "%tz}W`m{YXIugho57^Zw~H+b4PA$?e$M3>aaa5KWcZ+6!Uld_^Yv<k!xOyW<jGgSl#Lt2IXUG46x4;~FEFGq`?BFwuw^INgI1Y(M"
    "AhzG{r(gnQoFIBto$yDJTSno8rO!(yMD=<ZCZge~alDf0(8Y{$383bsH&whc!fi4=8s%mI5US*a&jO4qT&8eT4Zwz5yU3sX0Q`4R"
    "<0oeeS}Qqu;jKRYnJ+wbFM7S|#5*%X?DZFGLBWBrE3iy^x(qG$h@h);su#qb#y_qt4($KPYq{_&B6!`E-"
    "BW?SZzj1p%}E`|n_<2C4CX*HLvJvK&ue5tsXX%Tu{Me2(hW7gOoN_<krengDhtkE`@J61nP*L4#7k{HX4#6mVb`0If(lzUmBdg12"
    "6zY?Sj&a#a6+#y86;PZZ_YKRl<%ntJSJd$KwM>Y*X(e=-"
    "7>2+lHzgCJ5x0YA);mtU@F@J*P5%X#KUpK!~#w>(oVCy8)w}b)FB0LRywEiGAPJ+(+$%4^4tTgQ)>kkJLuq8`rzgmvscv~x0*x0p"
    "=aNcr;&k)F-w=o@BZ+e0?-"
    "r*|LXqN6sfeA+YG@_p+Yr|K0)|T~WL&*?!H|#yIcwd@9wsxh<<|Lt$yAz7EI$R_#XTN@Hh6G_+*arN*;;mEDle)1+FMcLsug?UF!"
    "!64U6pkLpM^f@cb*xq^cG>T*4q9bbQ8pd-1=CrTV2-&7IB>;oTj){C<Qe@wRf-AM(ltjw*5^92-"
    "{TqJKv$#=%||clUbzNA544;VIwit{r0+?;I5;4yUSo586G!T*uKVI4XcJ4cm&B}sAoZ74m>$$AT-"
    "|yJuFVZni!QF1m(GV^qjLxxQ~|F0@YfFLSu-"
    "|}0+`M6+0Y&}L$oSPs$V#Rot;#q(}ht9z2PUjD9x6S=S%R6ZSzWYqvXUg4keRe&1j%wn%CH!+`P#t?I<XF1@DrAcKv7lZ{cy-"
    "I`8B?@BjSIWf<j@GbGF}@5MWFLsKqv4;TA*WA^t>0+oErrG?sV2SBGi=81I@22DuUq}?kd*N)p{@YYE@RhIQ8+iFO+yE+x?cA`T1"
    "JqWIVN>VAy6#0ZB_V?1X3tXkm&mfU{YeY2ze8i!2)f^K&j)?^@u(PEBXtYGHXT)Q7^g<x64O2tXtpT7}67FdzO?f6D+lV}ktNOXa"
    "wQvx?fDG_^f^6*)0aRwL&0bzAOi;ie?d1%oAQ+xt_9E);DtjYNK4Mwy1SZG;f-"
    "^&g_0yOM9+gQ+;lGTRuWK^kPCe5^P|&b(;us3vL)bFWFgg4kDN($4&F1enuEUl%>h<6(@!jPXJ6a6;yG9ez7u^3cd%imX!HwGg>>"
    "SPxF*q0{-*9BjT8ORgYU4JHK(bq<PF$Z;%9c#maVAu-DT}s4lwA@ex-4Z8N&dvr<-"
    "KY|Ey$~mq0u=o70qnEB|CU;eH4nDH4Di7yf4Y;R$0Q6<sxA4a(~70S$`Hn4vUShn8dKo9uYZ|pv2f`kF_^kBuY?))7oZiGAfuTy5"
    "@7lRZDJ1|M-k>_QUNT+2cf8f7Je%qRcm?I#uc%({jM3KjA1`@3{9ZFl{sNPLCOZN5ELDpwt_Ec`vcWP<S~^x=u3jlb-"
    "HfJF17RJWJfIGuZu6D&5i0UupkGcS6m!<|%kOviw-Gi8*)?;9Tnt7RycmhYEU)aW|pqVAR%=8iqd5`3Fk|5lc;f$s{(x%w<E>;d-"
    "<^Pe3T=J4Tg+u3_Xqna0$mNq<>mU$W!AUSHGiq<kX7ZT%V(O42_8d$r7!yx?T?-"
    "YtiZMp<R63MxQw($lXyI+_eR_VHP%qvX06XB#T?dEqhJva%!YYZYfG$st?2uX;@93)6&B33c7}Bv1p>cB6r)zALw_Zu^dGqlTMz0"
    "$)r|W~HE^RG0zgtk?S`m+s+QLLY3k`7{o5QrKw=!X3^17C8VXVn(lh%9`DB)ckHtON?H&x!R4XLin7VM64Hh0=k@L@|H{sFz!42o"
    "MdMld?IFtLmdM^e(fX8TTg{S;WY0>(QX1lGM4!-"
    "m&Wf<nOxzTTHg=G|Ayg8ZaY?9m%!xMDp*Nf)t9KxT>?Q=G1#`3LxE0ht!%zW?4NP=uF)oIZD#?;55Q*fhTvI`FPCA96Y2`0yrz@Z"
    "rS~0pPuI}$iYdJeErunaMzQ0k&bW`y#21WPsx?G`ZpyH>Zsw+lBzg%~j&N5e!(`PUcB3(Ghi~1Z>Nk)n@9fgzCL8pSFY61Qmh1%8"
    "<sr@zvc)bETj=u-T%WgT(WkTWSt^ip<R<pr$_YZiedPVWH^pH8Or!f=4YW<w7RA@yhnMWnD?-"
    "hrxsL550{J(N>h&JrqE#ggiYdjXxq~GW2xxTqejX_{7!SA2^V7bFN+QB@Vtbubt|tuDeA9CUpBXDuhQhTCDcdS`%j6E|X`n#Kayx"
    "^?`HXi3O<99TxIR~r-i7F188xWD-^YQxp~*iwdV)0iO*iNn=^5i&?}pV_v;2dJGGv0+@W54*^+bU6yRHAll7o-"
    "n$xKS}(yn+xM0*w?PsgO|&-|8qGJ)THVhm{y&`TmjZ5uknm5s+TnP(@;Ow|w9Usrq;Hib3N2EO$kXKba+4N!$w&0-"
    "RLum>mb7yELQ01#hvZ~STEFcs<U@rj#DKxuTgza!}TYQ5aMqu8t2X9t$&9?<+2Ms+4Z;4Gx1ZmLcgThm)A>;E-"
    "f58&9^XnTTr?$mdiV>aWh$hK>sMz&LecbkLpWKZE_MH4$4+w%r!>+dJj>dm)tp?lAEZc;5=_P<-"
    "Ns54h#MSmwfxa{k}TRRf&q^}f>j9K>_GoRzIFz@sbO|K{<L4F!7eIW-VOI@k0EUVw<SUC2`r;jM+`{UkO+Oo4UKbx&NL+llKTdOC"
    "I2*je>wQN2oFz`~B1m8ZqF5stiNf6(yh$vzKQDX*1R>50uvt6p57j;XsKBDQi_Q{x}K((RFJ_!L?CUs$rBqsEQ^BE^_u(IBL=f})"
    ";p%P#N>{FYE4bG6zb0Gz5C3pdx038r2T;~|DaL3thP?)=*LA#1**Pi5P=4vv>q5_isw2Z|2*V=2THmc}8KHT0HAZi994L&0v)v!P"
    "5?|VvWsH;i>MQa=BN=cD82fyu75ppDq5W>M-"
    "Q6GIP0y38y`Q)8Rx8~*duq0Bre|9;}d#y;Ux2DZV5%k7<KAZFq^)JlKXMKmH3dr<k1a*y71ZFDARNwi8E;@FWc-"
    "+nadP)I`Eb`j&qK)=>Z#=sazu8ch>1L~V4f-s;n64e@7o3Eyx-ae83mANnlKW}AZx)z9JdUNI=Nl;;o;Z`WULHT8YT5o5f3^(x%x"
    "*r$GhOa1sRQFRU%8+6lV+Z0mgaKk<B6;t(ot!NOKy$EL<Ape{-!)1cgLQr5A!XJ=rw_&+)hb?n-$KL0bCRnYm-"
    ">bU<tpqi`Mx~T(sZ2qHGr@-ANI0mf`LM@7$Ag=Q%pESQTl#c9|b1C+l~xZQVQVV)ItpS0Q9wsMCSe@I-"
    "X#xO9$mNJ$XApTSPVU9knH(u|aN1VjDwULI4TAb_UUyrv}o6|*8O^_!XK44RQ+RhOXL=}el$wVZ1{p?m7uCmj|T=hb>Z52`=yj(S"
    "O9+VaY$yk)Gow@9$OrUFxRtW$hnaW?H)X)Ip~ank5xdOaqG3+QHzm77&#ve@IojXH0iq&%%3stu~0>$NedcXp%33;ZzsY!D*-"
    "KJGPj@8qc^9Z@p@-B{)Y&pVkN#mjXMZWKo+u2e+C2`r{M{Ol!fXTx87El0YpbBaZcbqx-"
    "e^)KT$GMqzMyLAv2LV#d~B?g$5w>UhYo;^m-@j1$_toQ#V8jP^z4GZO-|DegEI+JiVNl-IT=el?kfO~ed5Ru&PlZYnDLZP+^K}*m"
    "yV2qT1!E<j_csFD<t{MH0z7igy7i_B6=m(QH&}Wv1xd0URBo1H6pJU5<;`IH*9IEvgzK{J23nxw7sB-Q-xxEc>hWuMY5^suOMs)P"
    "7Mi+Zs?29CL9D>6Dyc=IyS*!sr=0q^G@M2H4Zj7h>_}l6u_bC=X;|7i|tKAlW``B;RWFK51DRot$08m(4b&NxfE<@ew{r%JTE-"
    "{=g0VYO`+(XM<YLy#a3Dge}TgMx(SZusU_i%6Yfol{U%jccrqgz=PwEF#biLH9#<~L_t8sHB`^w9Re750Ae6-"
    "fis=&SDINha7C+xpQ!N)r(Sh&`i)nIpJo@W#kuh#1Zpy(Q6=3}QJ)<SOg=iF9z6ozsWMy<|4Ro$&jedirm4!iA3vptG#C(w2z;Ug"
    "O?_%RGTE5Qq&(kFb(00sgwalcZ)nXm+kdRH}Bc3+E5xrmn2*mnK;n0X=$SD&s;fPsg2#qH{eQ2~>i~qI_EQg{o7L(kkn=nr#vJKu"
    "}JA@@g`Rl_N-ILKiJ;*DyqX*Z)0BDcrLi7g*8kcNmM|?tHeQUneO<+Zrrpfe!0!#m<6=NrF`eYykr=S{@d&K`A}bD!@c3%iD<+t?"
    "?bIphaW#>;y(Hum?=t+6<8Fg=o2cXKq6ae651x$z;w^*aq0i^2j7P3zN)&Q_)^czrsSLoN{_Eqtmn7(31t!GsDSusc>nhdPCI-Sm"
    "lmvp_wmLwMKlmRoBmx@NM^AxwuyZHl0<MVZ1i@iWsefTvlb;wV!{)@ii>wCx@%hhPTmy@bqhfTW0PIdjxe*g786xYP5{J!ENKto~"
    "f7GpEzsq3)V<(=R}9>W3X1%Z;47E@oJLTwa1VOtJd#><(fTw>c4uK(a~)O?DgJ$q85%RA0_XFDRqh1jRzAbNo3jC!HlJe_Msd+CU"
    "1_C42dN^z#&G%V})*R)Yy_x-u5~e@Hjd=!)I?q?q|H4<^x?~@u^!}x7yi92#XNE_{oO8j$gNgpK`vP-a?~pOO$oP=)-"
    "S^HQ$?Ce<pn>DVQr9?Auru_CQ3`iIMT%4bs1p1tLxG>Xwubbg36q7Es-}u#kqPF7e-"
    "2^YJ#0yq!x@(|Ss2T@pUVony6GYdRk*J41lZU$R2Vep7eX27d&m@KlWzSk}t3P1SA3LNa-"
    "NJ*|x{W9*X|Xde$I|EAumv8VkVekI@IMo3USsD3%I&15L|oT}i9R#Ddf4Ri7NVq-W-"
    "FdMdk6NIbAKe;~NZ|<btRaSs(d*5*(?ca&`DA{0*WS|?LP@g6`K}7ND8suj<`+MV$)ixP$Omj-Gg3$E@0^h|k*f|#^Hrdu1dSe$}"
    "9bmaFR=GOJ5;d!<r@n2?;Ug11@?4Ys5x~(U#zrqNrb;c}tG74f*!9zR1R<{@Q6$cEJ4CH5^CNLE-"
    "=@ZR%OOj3Jkosi?jCoGNbqS)GOdx#l+kFj^VSN%bY^2L)Qi0+9e<ULOZl2GH*S4lx7kGw7R}z{NJRWOQw7mtNvce?cVEW5Uh;~P?"
    "piVeZ-JGniwg+6wLDof(B3A=bSzqtfF?ZtxWCQOos3aqn5C1M6x#q2q4HdFH|$}1zARr$)8*xTk$#ZK?v0J6Pykvasb2^9uqT{?8"
    "_R#O9yurY5lv$lxGe)BRd#!Fm(O3F>|><`vfEqA95^l>v?4<c#>CqD_YEon8EYn8(u!0c|3r`9+3*ko6eT3rpXS(>l~eF=!EBShL"
    "mF)8<jcF-G4sT8sZLm0CF_rEq2XRFx0bYIU2G5ezT5Qbm{yt0Xfg*DKsIy}|6R{t(q-"
    "j2F@+nwG+s<L5MF}{>Fn!fQQN6mBaRu_hGF9oqR{h|juP7Hszqk{JTR#17n;u1&}yx(AKXs<s&FDp7gt|JE$v4irAtBw6SKaw?Ho"
    "50`^Tv1Xe?;WE=^`K!hh$<aIR|~;-46GBWVo4!A+9X;0;YhQafpQ`}=+qZJU&F!;X7o4zXH2p$5Ir2o-"
    "V;Bt`A0j5!j%H&P)Zt9N>(=*hW1m9GN}MRrmKZ|Aq+B_mz3DCag<_rIr2HR^!2I_Vz?89AXjWI#vb4}YbWkO|+i*SrLusUOd{H1="
    "%-;F-#5vE5<s#7%5u`y?48B-"
    "{2a1?ZYYFJ8+%y~SO+`MuJdy2g!iC_^Kf?8@0(;^AJhjT!#_5jW+Y6phwl8xMi5NFO4~==0GOF4RYZopbGZW^(6F!jrdoCPuAy@?"
    "s^I+}o6PnF=^{dW83TeK=2@s@RWXa9QG~rIZbm=B+cgd}u+`n)J-"
    "@@`%DXJXUthnp!EHxu(({p5Lj7H?cB+bfSQfM25JedgnI>FeqEIQ_Og<SKE;x8{5Z2Qeq6|v_LRGdP|m2jyAi;zdUKrfd5*SZXM5"
    "O5Xx!?GhY@L5A2gKjAp<3V@;fJkBH0rpV{g<DdAKt6h}UbcWZ;)Opl`cCtS&>@;bgDqi*UrIMUBqXh^#h8f|4HB5Wj@OOPW}?moZ"
    "zZzkRP`gdx`p+`1vUhVVixIXgLlAC18KXu{nS>~wQm0jmO?ok=<9?d&Y)v>Ox;H6z)Z0!cQY0Yxm>|c!OW8lg}7hih^0h7^hiLBc"
    "1d$MEk;TA>3GJ4>O1&g$fp*Du7?4_7Bg=Dvw$ddVYf7|6GS-7-"
    "fAY|YeK@WC#*6)G!emnuey~l^jt<Df0;cOKB!{rJP<OZ85A5NSREYHo>DI_+aXB6Ub|85P(1WPaDskP%cJj{2VoQyNJr=PqS2Dju"
    "Mv|ye)ovLixI>dxJi57YbH50VF$pV*8g0`N)tTumZb#jDg`=5R#3t2v&)e?CUIxNQIj<EmyYSltg?s0U`V#BNVFvYF{<iZqd@^!-"
    "F`TdTeS#(H}fC1>zpn^J-&BdG1h}z)IPbs<a*OWX7hLs%E`-"
    "LMW7M<s8#<&C+M5qU!Wo~H8*qA15+}qUr!neZT#q2#&FL8BZh=SgmiO(9%3TnkiDP7|ULJSvT*y`S7oe}3}nsLwC09wxEnQ!ZJXc"
    "?V5>GQXbCWL;OtJBG~O*F)kKSt!2k!!D%gd$l|zR_v-Mp}xvUmon>I3`iX>r0GsWNCABtc#&o<oJkq+azCmmG2HNZlc7qCXB~+6b"
    "1I3$uykB6^3w+9~nwBgib2sI*DLbNK&58q&C+(n1T3}e5$>Pkz0}F%-"
    "~(gg9!%lIs<XWxGy0j9*K(3q^_PWGjU>+=myTKHDm!yXZ&d2^4+G?V?xIC+U=*cyNM2v0B%NddK#i35Z-"
    "l0_rGr@U=oCkkC{w>sk2f!?~O=TE$9=MtO|aAu&%i?^BkQNM^O(KkW-"
    "Mc4J6L39g$dj30L!oU0-|?1qjNT$7&*Djf*ml&7QTs$^-zcU~(##r-SnLh)Oa&IPZBt?9NGIoh2PayW{{rV`EeI*Wgq-adtqj3D9"
    "IC88|J~PM_ll4w&%?QT(aVNlO_C^<)DdVX###bLU~G)$ZFa2>{H(v%b+xl_&GgARzJ1lWLIs4>yw~d~ZV%I4w|{!J-"
    "2;_hfo7e8?gk#$(Dyo9839bC>?+hu~W=W)4`;EH$GU6s{f%<(2w<PVB$;pdFA}O8I6pS-zy_iUpBSr5R(ZX^wyZCk(x4>r_bhs6`"
    "Oy?yfN2AdU`j{H`$M%u-"
    "rH1>Q|eTa0iG=4s&1&nKfXx7j(^39he2Z&K|Rh(xagNsKenv_5G*D#fm52Z~_XzsuIEkvBD|oefYr^9;(hv8Jh)^$iFunt^ks-"
    "F`%kEV&sien)vqS`oC#mb1+61VA7$(C=8{FR8f35pSx9>N}(~?2&MgcTPfKDj{<9bJ=I}x5g@C^yEzIa^@*OPS7nA{OUY_ZAh-"
    "<)^O}SBSZ;&na7o}cQ4sXRFqXCCzubPk4|icY41`A-O^5kWS9NX{sxl5uBy}~j;8pav+)|efQhDO$ZqVKfKDyy6?vS&ZR?t-"
    "#O`3VlOOOj+tsbBrJ}j+#>{cR=Sp;7I8ybM2xWi&8wz<}qHm&r1;6TGkLh<bo4-y>3iEJyemyGZ@|QhdB|!vL%nV0-"
    "Gu2cwv>%BbTJH`+qz!mN>Ku;>7VW>#8r|<dNl*^=eZ8AxX5RPv-"
    "%Cv6T_uFy70VIk@;wi=C^>}O04~eudxYy1Xw@|sR*!&nV}hb57H~;8C}}@S@6{TrJt}~dYYyFQ$Pq;II-"
    "9{0U^~X!wB7M_^utE57suZncvb(H$-O>Gk-v?glkCN6M+tIFP-Yg76bkZFJN`_!$ufXyvDamnTzQJAY*)HqTSI%-"
    "8bBGB0uf$MGw*;RH0?)t+lqG~U}HYwClk&s-n(B1@L&J+&wu=X|MB1dM-@bG)<av!6lHlxgy@6Kd<?_Z49>Wn9#i=>x-g<;It-iS"
    "l1iGdyUl6z?_STGqH)0%??ht^*VY98l4P$~?~`3)9ZPX|GNTtx<$MHuGq9B_Te?eFmtwj8@`k$t@=QNrXsebc5Bg$04?{Je5)KLx"
    "P5-"
    "n_F8SRSVQ+s!qCZVWo;NO%Pl$75zz#gryE1MP4aJt5bvhxtrYB_4z(3k8N#Ohjdrgtw_v9S~GMNVPcK_z0o=k;3OnWCPoq?=Nsqy"
    "hwqpGR=brf49oFC!XtS@%?P$8$~4k3eo;vE(;WuvaXd~lxsp6)+eXsk-RInabghYk^?XD?+~S(7Q`A{YpygeEgQCy-"
    "98?nU_}H%;`CgxE@*B4AXw+=&Vp8vd0#mXCCp%i;{oAfq~;)!3FP;1xscL_w`F=}7ecogUY`mj<-"
    "K2H0?{>qIdm%Llda*#9!uEp_&k#Z7Uf`6g`T1l_}Z%R6XGZTz&R{OL-"
    ">Rt=IYLAMj3;`epUZ%MJA@|hPp=w`P{OSPnez^uk&G^pQKs@LIoZ>2SRq-iR5fUQ+;26k4|M4EL6ng*r9IEWO6Zv`et?{oT}_7>J"
    "=DF`twe52<D{aE;~aizZf0tk~iw#X^!BQIqPm2f6k)rBGapp%FiTRIQp2m%Xuh9|8iuV=Q7OOAhff}%JcNk~A$IsfEtXEObmoS!l"
    "0O14OZq1t@gzg92k2dnD+X$+(Dg9=8yc8>7bji%t_A8*{re#c;81fop5{_Fqz^S}S&fBlZkGG$rpo@X*9TimHG<~k<p-~l-"
    "*k<Vb-LbrKNwQ7_o^DEQ$kRy90bd{@``um!hko1a!yqIdL_Y^rJV4r+m=h=w<G(o|C$4Wh3dDgGkxt_|!@Kml9z$k0??U4|kLJBp"
    "{+tY?)yE)#eSjFnVHP@%<GHYG$F@7r}9M&e{>%ctp88yW*Vb}-NchIM2uo${vgCx(ya4iWSU)sdz@m)CnAimB4R7o~6x?O_c7`}-"
    "wW06c|qf+xrgeHfMsG7Guvr5N3)3s?z6%21C>0lrIIAFEoIeVwCX-Xr=Si|C{XKOMS|3Da2k<R6ne1hcr{R5+%o_I)fFnSG4vIcQ"
    "9Ox*TXR_iq`+F1Rme0`eRhts($>h|dyq)g858fZ9Q??{HO1cgHzCdbsjC~@itz?DL!O$wV0`|mTm5^Y#dp!?QT8N>JOw6}dvV(g0"
    "NxAY_Z#5-"
    "VueDiortrj86?~AxoX7rm%A|{~+5OHhh2xZ>y!;MYwn*A&dc(#MYisJ>#h7ec4^Ko{GtT?eS<+j<ry;6{&{&U+sp&n#SS^$B8%Uk"
    "S$-!YcWKQ(vDNPP;%L56ND%k2;%v^K2qKW)f>n#4{wGofaQE6LAT6KLI!L7wTTk$y>X-"
    ";puftl;KKNvk>w@7N7<hFP+G3Lt1`2b^Tkl6g>xD>uw>&xR-xXQ6}g9S&PCNu~j1!&pBPm3X2voyp3G6R(dUZf(eP-DR&0!&a<03"
    "8HRSe*A`VcqWyxgaeVJ5I>0v>!>c=Wb4<YSp7bpA*ml8SKe4|)OLm)B=SN{lycRK5}ZiL@;&~Z!K$GA#`$p&Y+5t{a$p8$3*MZhJ"
    ")+-mtNhKArtLqN$U<J^FFEju9AT4StGZ|!L69O*a!*KL-1MvvwmCh(d{V6-"
    "p4(b>kPw)e)8P)odW`l$Cr>XKphyigJRQV}5ERF(e-mA=VmR{5%zVf-"
    "o}9Ebf{FhufS7H`@8s`uBj1{1hpV5gai6h^A`giqhh<%;6a)U||0jXQ9uW&?&E93A45<^?LcOqC|Jp1G4`PT>OW(0(FbRx1!IeAE"
    "S}03QLPVNXmxRm7;tF&A1(ZF}sXze`Fue^%cudq>NKbOWt9YXSv?Dj&AItOaRb2}SL+&xcR$I~I3`_Xg_R%I>DrhM?5<6y_B6ypl"
    "!p<Qw_J3+kZhKWMgrg%wP$Hr#XIfZlN0JbiqteKq(fun!Lbu_6@A$FDjO+Km=w)Y5Vc;YD@{Ve|*A%}#Ds!@HyZgYSvX#;sGfE%d"
    "TcTmeO0;NOH$FyH$<A?7b*=yz$rhQzHj1+gN@5S~Z682%z*KiC7{c&o3(#A-"
    "rj5{B8M<qfVQcTi?6q9z&9`G62h7zIwj(SoVJt*5lOMh%7+&1^-"
    "IPZ0sStE~;<3u%xkVDAs5omu(({p{S}y4@P3s7{+yBCNDtidXgtZ8B8s<nZ(rfW$?sxBlZz+u_hp3rteJxIWxrpE=6ZdBwN@DD%h"
    "j8h!#a*hppp*CJ*RA2bMJPAnCJ<_gKnCmhCepMRuDq*~bK3mq_uer*ga}xRjEq03%qO#|KC4%G*XtVR>s?6kRVId+c`|sb7+S-"
    "@P1=*qdykX})AktF)=)cU>HOxJipcnxVaTb?OCuVyWm7<6S@n<Y<hp8G61S2xi)(8H0iN+DZ^i__a7YK65L?zBM@HiU%HX>NE7U7"
    "C%@XxcW{3Bz!%VWQZ!8?&_3ucxm^pR+gR8Yn)4Y5pbA=`OXd;q5G*>G{$$y?T=_Gew9d03fJgM!z>8dN&@E8SsJZ}eCa)Bgk$N*<"
    "MCMbquW;10tx=_8Jl+=Jr%S7FKOk9~1(AUTocn0pIB+?1Zxu12NCsNpFAJg*HJi$kTt6J*0=Z*ErOrdHV@jS(0UI)%cS&KQWK>Gd"
    "U-887-<g0N6ubn4-l}_yUSm@+GcShV%c9q_NE=i!+-"
    "zeKU<iHq>29q}3khphSX92{h?T@i%1UXdt48!Dx$h=wDfWqIUQ&w}aXYl%;y&^RYy%U#3i+Iobc^T-HUc*4EMb~$HUDm-p0c2P-"
    "eYAtw%2B|m{1$a2Ik?Y&I<w|vd9syk{XM{hC5Z6oiN7_s=l((;f9|GpJB(<8mn1<xlR}kjW1CG>tE!TzuL!CdxzKKpE$C51(Q9<8"
    "lGs7A)zgi+Ss@8<9OKODPsoh5;^WWE4EqQ%1HKcf@O!xGnBCkrb|{s_)_+Y1)EoK6F(MK4X*+)V<NIR^5p14LD`wHGC(G0o03CYs"
    ")_5*kX&0NXyZtP?$$1WnE4egnW$?Fn@QiNPt-}k%j<8%B4-"
    "5+)#L8gCf7DK{_7m<PAMWpHm@PRZnkvtPcv5Bz<V>!TJ~IxR{f;>s`A>#GM}`~~4kNa<tfov9S5N`mxl3n@oLQ;{HjhD6(?r|xCw"
    "a6V#2NvM;rx{+@vPI$W$hKW0txgOP4=8hO8Ir7qwOJo)g;X@3wCds81R0t)0w&Ta0YN&5`UCTtv;5L23Jb|IXT}XJzGuw?%M{kh="
    "ec{+Sib;W{&#!xtV(5Z(7EBhjwwM?78|cCyrFQMj78}^KPmwq4YpzpGXQQafz4qx;`SGThvCIkL?=W{uV@Be6r8=X}RRa<@-"
    "L4FA5BBj#f3Hj+1ljuM$9&w57@B4k^*jU4veKI?$;+=}(E@U0dYubVl!pMY)QQMu5IZ$^DC2&@yWn6W?u40}RN@80c@^tGWs2${Z"
    "G#r%R0AA!CdW7`1yKt_hRg?}Ta*G>wbFE5pBF#SqeE88yf)9E`5qZ7?ZkLSX`w%Y0#gi%XIe5V#vUMZw_VK;qfofw7(06UFhc){h"
    "^{Vz{M%J%5<<the59N{p2|WYJVw+xQU|LruBiD4$SldhST2E}JSJYPA;hmZk&=#b_E6im;_IPV%$%nwA^e$FRC*Dl$?oPsG7lzh1"
    "L&zfuU9sO>G4v!-"
    ";9vi|L34HZq}L7}Hf5>dZB8qB;8ayaVJvO&hW3+uAw#hL^FN?^D3<KbU%3VIUubwb7Cim7C0g<sPI#6i%O|DJ-"
    "%bXyZ%Xu+y8I|rlwCWGS$M?tY$$K_SVz2pTR>rm|yYZzcW6<`N4UI{s*;5~ZexDpttvY!MuU9+_=4iyQnpQ4Lr)@;H(1dr-"
    "C%H0KSKnar)of+da3~W(DKfy*Rb_+)=x2=g~j)GS^2R@PQj?`ZB*Yg_OZ;ZAGi%a2Fhgzavc`{BL&;>Q9FFe|ZrSJA93EMp3>_$~"
    "968Zqm*n@W@(i)3++_$c`4tS;yzg138xYFNxi#{qhMJG8uVJeV+@<aC*?)=B*%1(8FH1J~@)tnty%|;YO@k>+8x1yT~{?NZ35t9w"
    "L#s=xmuh(6c9Tv|r_F*UlgC3{UMv2HTRElmgAJf3#NNPfHRT9<{0r+!{-"
    "z>%25<{#HHBY4?Q)?4SY8zq`Q%mt0e#PE%?8is@>tls94Y31;$E<BnrTTt9WQNbg+gJZScrOMa_~q9Wn?77?32xE#4I#?_Uwjp9q"
    "B;51`b17TJ0LUVuI;^FFL|nWZLI5g9M!(0z0YVX!=o@4TC+1eng^pqzL+46Nkvd8D#+dl&6{-{$Jm)+8d-"
    "Z&0P^k<hnWG4cLHnZPE@o8x4WWr>15C}b>|qEwk%}s#Fv$N!2P)dknWrsxNQkA_jcY?<(^zgR%HHzvdX;(xb*Svlcl#%OXB@ZJ@r"
    "wuoANW$Pa9SJUYUL=FnFxlqUAmMb1%@sE1#-~0YGk=O$i_t<UP!TzJ9ae-"
    "8y{J<;v5i6Euk*{C!+@nfF{HC{L=eyIj3>Wu<`;EaJ`VR&GrvnLkn*lPdz=e>Y_oA99HV@;O?2ZE<K$lXm#(bPOvpBf3J@3@yyp?"
    "}-hZb?L!Qf`0VT8r~jMp`0r9VHy}zy40axw#qdruS8HlUy1`A$=+k#gxndwn*aBk0_&?@tO${Pv|IUd)s5%6RS%uTWJHHNW-"
    "XW`D$-b|VBEGzG*JeoB{(|b+!6-h$;NZyd9zb`97j=NcSe`B{+>D%H)qly-ryCsM4V0~#-"
    "<Yj_1!jJfKca}9iTg_e?Kk*`!TqtXMPvx`Bw<cZSGzkdMs(}N;nZi(8+dQ%d}7S0@^Zf7bHPXfV`Bxkk{Yf2t9j@5lNGbE@`&?Rl"
    "lV_r;RB0{(~yQqx_`u>fK5zJDWH&tvV;^kRvQ@bZ3f*aO;$e*I?c`5Iv6}em}=(`GerPLoMp6%nn2RiV@VZwGs;6H~_P!^l;u~MO"
    "I>8x8_ExM-"
    "+LilgnJ3dvIIAS6<8BcHCtLU^+>iGpNEV9o8f9;GDs`^+dKc^5%L3%Rb8vu)9IZTDl%FNAZ;WzU+bQ8{WAX%xpSJoBn0<MqD84Pa"
    "Pfd43;DN(j#-gkpn0K61Zl(+{#xXTpBWG$oxk)_Iw*>2Df{h)se}af<BIS(k$tdEi=m%TZ1GQ%?<eWM}FXpA->szZt0#lYl0;+-"
    "0kesv4si1=Me}LTnfw;EI4G~ee9juVYv#Hv2z8;t2CUzA-d)=FE$;mkDF+&RO@n6xhrU=tMeAAAst+-"
    "yMu0$32~q*KP}d<E>7Fy`up93ZG6mtGAk`(qOSaydl}9wJHEGza3%Q(1{57mTf3B~%^ZpK4$FEsM&6i$_gO95D|ikGB)h3TJg8JP"
    "Ky0-"
    "xu%Zl6(B7;EuE{Y5c%aBGtj1L+)HC9ReoU!(Grpu38ZfOD?wxs+ny2SRfy_&|^KCB;;5O_4+B1M{L_>lKZ9v|j;hKn@gPwumIfsr"
    "CS0RN!0`_{epT>qN$NSn`41Ll=DDj5>S3})wxI=k+vzaPtf}`>*Kicv(9G|LPi~58$($CbJJrk#92+!@~fLgJCBuRO`^aW5FGTca"
    "ex`c?z>5}L~=RCS=lleFUvySTg9$`XjR)7|&wI(baBs41rZGFHAU@#>B`fAOnIVnOe#SD+q-"
    "Ll+BL)V1D99=1)&px5Tpxx98W3}!)Q|#r>clXV>@GF5VjmvtTo{KXLEa#e+els@r>?e?>D)=>&4^Rx%N*|vzsevkqTHHtK2S>an7"
    "M8lIfABNW-qsuU#5IMedItaImpLZBfJEzxrFD48h?BeEEtkW(bo9_!HH80ZgHL#*BT!>RS^H8;R(B%J80i}efHnFUtr`661^qs%C"
    "1rmw(jJ~jxBwvM?;Piw*PYUQ-6~VM^*-S$KGAU3oA>*#-"
    "HEwL0QqFDzlnP^Kk`cTJo~w;n_v~F;UQ^a{diXQy*VoacLZ^7p8}9rZ=zuBPnx4ZMSSIMx+fLS*>@*@>@u{t8$Dn-"
    "$yjG)_e|><533=}aV{me+Ol7FL7nox+QY7-"
    "78+D3?tMtEZ=75#HVH8JHkNvs@CMjk))qglxyx$Ze18^e5|U(R>|3X%68XIQ$Cv7;J~xZkd5?O%ZXyISR>;-xXGGzde7-"
    "UG&wGC)5*d-o)U-|#=s5S{m5AR@KSPp0xOWaP_Q)nM;Re4xeQ3-fYbRILIHpypL@)tHdmFpwzB20-"
    "kK=T{I^M_4$o6)Grqj#Y<Pyn6Fw^d}+7mAS9a2YVza^7Xi9mseT>DSA2$b|E)!ksdLoXyoT78oZ<<*ql$*NuJ=zg*{l59b1Rs-cF"
    "W4m{9Zb}Uy;BNJPCfa?1;beZCW_W^euI8f|3LACpp=Q)p$zb4jk?2i2&LyWFRi~8Z?QT7ewr9;ykSU;VeS=Jm^-"
    "(ME;_L1KIM<0fQ#;e9BcV%6LoSa}y<K72c5qpAPw`P_yjiG7(<d2Wy4h>};<#DBsGo21#oFyLvqf$D<_v3{{(|F@bmRM;(9GX!#7"
    "4IB`xKvLi7_o4_a&a5$s0Wae@uyJYBl-sJ{haTwO(1BZb0)12Cl)R>9`e;9@Dj7O_paarg;Z(Up7gvH9I1Re)ry{ktiH{iaa)#`W"
    "ki_7tHLxkFQg|-^@%-d!cbWr0@4}u_P`1c#2Yuij7!5|2d=6nscqNfL$C^BDE9Pl)8sK^0FPRj1W@^DrPiEiMu!kmFrE(-"
    "c)r_(<3v8!;JWOk1=#|QU_d$6Yzenaew|<!<}0?#cvBi0x%|pR8?|Ufb2fZqZs%Ui55R*k6H6plpSK#8;<Mm2d}KM%a90Va@RWzB"
    "Io8JOG>Bt@D$&=wREfN(?dnE#L%$3(?-}c7zv}~%u=}-=S2wJtGXOS&XLVw-oWQ7CO_#@WZt3Lc6>(V^Q5slqNi<b8K=yhY|NvHl"
    "p77q5+C9+4!|cX!3u6vP<B;8@4huTszrWl8_(G%aw8Yza(lCVR9y50R8`8ZR1<yGr8m5|I)Q_t%Q2h2yV8Y!&)+#|lL<k2{IDSk_"
    "SQ^gs)S9jZSx+N;C<l3il(ICdqlo7x^E>!rxi&Coblf>^2t2ef?kb>*X^S~Y5#ujmA!ehSgm-"
    "BKHZ#MQ&UNDuV#Z}X;}MEr;FCr6u2}5acgaH--sF6XBQhT+q&e4-5c_?$0SU-Wcbjmxx?lVy=ep$v0Iu45HKUk7*sELP`Cn++j@i"
    "+O(J~lmAJLkg<2s>g|nrO)EbY`_?5kH!7}g&lzX{rr^5pYQ^uX@t(j5BOYr;Cv|DGZh)#rU&A?r`mo4X~<9546h)}vDh145I3F2-"
    "WNbGI7k{*(SX%79RPB+5m?fV9GI(ztUONUGCOHA3n$30xS2^%sH<jU^ezp{vZp-"
    "J`l^4h&Y$Vh2RkA6QawD#n4AH6xREWOG^E%9)Ub?c@zQ&D|OsW{?f0e^YKvk%^6&+J8su(g%O>*4^yX5EKKm>Mx^{(VA5)YC^EXc"
    "N`vOK7quelw)VAhmjT=Z5=|l&cSH7cxzixmu@)ztl%(34On$i?F>D=V2B#;m=QkP;;(W!PN=*lqJeDt%>Om5_D!so%?<Ws6fcJEF"
    "W#GzCQNQ2=QsU_4hx)n|S*rGvQv)a5p-"
    "$b+O%u9(RU3PvmE%pCz42QWZ}~7deGOd)*i`L+ukj?P4O1Br2|_VhMuF$?y&kaPj;o?G4+2`MM(tJ1ofa<T^LdCPM>V$TH)%DBL`"
    "@TVk_Yk7IaqzfP3r%_gcAhhhFECPOPSP`)=b;D~~&95y>xD960@s)&)r5J6E)^J9MEKbKSZmo-"
    ";8lcBtqB%whd1l55s1yeex^kq$jbF4)vxMo?vv_!Qm|6=qdWy!rfs|~g`=~2er0Z%?1V|1&Jj1;%Z=wO;Xb%uKsT<uKD*mz;qObT"
    "bqRTQ2s84HYBH?he{JzG;mjMM-FRPR8l&@^XD9pXmUg>Zx3CJQxiCZ&U~*q$}fD=ZknMwbdtYUUOsm__+Avzn^o)+kYWjOOjMoTs"
    "3f1=vdXNO<tBu@*M4*JEao#rM(Rm@q1mC3D$VOo@J%{tW=7jgT-p`Z0(3<sBTjB9i-"
    "&UB|KA)BDjxMKVJ%o72jbAz7J8>l~KN_T;uV!+q8c`#T~f)2*LdFJf@r5?D=+54hC|cS`V3fR|SFDf?NQb@1d%oxsTlVbXL3m%rB"
    "TAV6iUZ1P_FG2o&g>{38ZO%+Vum9T5uj^5UXH5yamWY`DY?<%m}UK@BWy#A_<(>*5SU~CBRG+Iyj>DINRauI$FYFj{2h83|jH(@X"
    "@JL#BEozjP^qvLP56V2HAa?6Pmsl`yJmK12$Tqp|ErwdrCv@J;3wNnPenz{qV%tE>0Oc^3@&P1GALOoMe<C%>1Mot4IFN`y*{hC<"
    "aGy+jn%Y<xg5&NS6KyW17u8tASqy#9zyLR{42_SoA+xI*}BGUJ7nWoZzbt4H<wg;=Q)TEFUbd4`--"
    "?Dn{^TS|+q=o{ZY#1T(2f)7uY0$mISs}43R;ji7Q1TCWyP;OsMhPYe`U()`YW9hkh#hKwOZNHFDeCAd@hT?{poFiO{;JY*5PQ}ML"
    "I{>*(axL__RNp1aa*S0F>_3afI#?ll985>QNZt>rV%p^8+9C+YCf|e1BEa;4}N}DCyXvZIg>r~g#$+FQzK;GCd-GMb4{#z$ciP8A"
    "i(36HJs5UluhRL08lwP#EwzX#p;_Jlp)681A8Y-"
    "n)yXqlL+j*Wgv*ZS6$r`ev9b(p}%8DWHAX}d@jd_mN1*)=BMi3`iTk8H~UXQ=2E}d{1O`|HP0G`MSQntB&cs^qP}O~wkphQ7h9>n"
    "=8E0{YuEM|?2hd{oP)C;Y8Rlro^&B`BqLo|1hV4F)bM6aC`UpQboOKV3bfQQMv*+}!}N1uGgxBRsGIFI8Ds~?;tUxMiK@^}e>PAA"
    "AsaN}rG${TT@EZc$x(c))e~CAfHey>945La0_doI`x<esfq>{k)^NM)Cx2qDAsK%6<HODy9GL<J{MnR>uwv=K<O)wBu=mCQI1vEL"
    "-D!}?M)_Su05dgQ^YGw0yk~6UeY)Hc%?0%c@KH785-"
    "jl=dhp#G;wxyRH*pYl2PME)+@SofC{itIj+9r~19pB)#P?HiL{xIejE40z#_KL0A8QTsNmo=Ys*JY7fJ2G|IAHIc3N?uX{|6HCsD"
    "F3cizav6!T;7N$n>Cmy2Wq}BtU2>)#4Tcc!RFvB~AYPuHI<NP_*+#Ex{j-"
    "0$TOASy;O+8k=R0fbn(KOkjA{cIKvhs<~Eq(;e!20YF<Bl3$VlD-7tf03lK1aCi73-"
    "**_!N5Gc^m0S`N{jtX6OH;EoTGI4oquDSp?Fxz2NdB&7axgE>e$q@Jp>5kIBsCl&yJD8Vp7A+Wo?Hta`Bp5m&`PX){?=`Iqj+&R="
    "Sui}JaH5;I1KiM2070t(fLRuDKW3qIFv*qSbEE+n_0)+o1-"
    "&SS<{DCDE72J`DGJ$wHEwn4gF?IJnK>+(+D7MP|iyzuvZs_l8m;k0=C<WwI@7+()#-"
    "77}xd$dTBg~wAS0L6yO=fBq*?k$OPIJS+5}F-"
    "gz!DxU3aRS{E}#1+RuF?NW$d=2p)HrkzHhr#5W2*$HoXE9wf(`kxfCq=Ok5YpsFrxO|Tq4uKSzdAl6KA1IOEniz!;mDX9mWS|+@p"
    "x<W}=aOV_Zl5$4rPg#05*3HqUSn3G4ksvV<Jyif?|TN8ci@0kp_rwb$a`*k!HERqi(RBHyln_q;$y;XO7Fdi>B%ez*+y(!d-"
    "Ot4SwX0<FWYqzQ?7$k#^=*YMA(NlNg3zSF$VrcUQ_=$g1(xmhguof7<*A2lW(vx($QUblMd=g6SiuN(4wGA0>;0of2}wd{|$X$ir"
    "Qzz4w%WaBK~Fhq5+?ZOX1p_D&Y)BHHpT^Gvx_!a1*FzqB~>pdk>%TP4G@T$$2_}=DAgWZ`2p2z6JxBbcyZxOI1F#>(*``VKYe>mK"
    "hQP-wA+Dg81f16v@}i4A&Efet1k>%13GTB#!8?wCk%d@DPt1iPCu9<KFc3M_tuJ3+h|5O|$rFwG-QB&{gi}vsZEC+_Y}^h`ovpaZ"
    "L^IG+uLUapsmdykgavdfD`uldWIFx+@>!)_Eb*+EGjFri_dP@okT{9xE7mjFSc{3ZA(`YxCguKEk7oa_S`6IzWo1Uvo+~-"
    "t4?+Mt(=X%b_Q|q4{uig`jrsGvWd&i@$2i4q&1QAT%+%ylaz`x|Y-"
    "OiNriVPxMhPWP+X<&2B{;SknTV^3^XcHcgGfxWI8T1NvXXWC22lw_P6-"
    "?+v<`W~&5iXVMl9*VSfe)R?MR?ad^N0$VU_vxKOzjwjnY9%>0hj{!?pT~?k!&yvNq(=&4&UDO$^$k+mw5LANbTP7<d&<K&9cz8#f"
    "Vzy|n39C9LDS1&UYzO@@$a$q}cFx+&n$GJZZN*}(TY8gonjwHXJ>H-"
    ")5<T@}msx3MI{CW%QQ(D7Z=m00u7v++UBi+Rb4{6!;!?+f8`0D+p4nsTy93fCmh5AjWRq7GjM=LPy&KFcXdb-"
    "aeDK|A!60J~ZnmMlw2e3@sqd+o#*s>xMFgQX{qjDpEwM{X*3Fs-"
    "wA9c!QsVAcY|%8*<fAzq(vy)L{yz7@Y?X$dK8@M_<ku+A`AQbG5gf3ul(tx^YY;|a6u~dXn#qN`WGAzKOek9kDM_wkbDLf-dCbH("
    "*?T_c*9=q_&xJRhk3hd_G%%J{)fVAn1hL23iers6Yn#LN9Y)*lC*{F-"
    "@;lC_aZ3hJzQi?@4imr0JLZ4dhE`=a2>knpa&sESrNh|K3eKn2hW3(48kq=$CDC|P4^<w$c~NPNF#v{lVy<;1%&WCSf{J`CS6HM="
    "k2p0_d)f*smKF=zN|?r~O7>O>h3Il;e{#qhX^S8r$Q#7y^E82cHzZ6zoJ78Xf_ad3J8@=_d(uR<Ac+LTpffajCE0!VHetX;Z@$d&"
    "U~&C_^)F^1Z-Rce@hcyie?hbC5^J0Nn~g>^QhOWa&7)(V|8?68vZD{*dAvd+<+nIHi!SVo-PW~P@Py9{<x6rJTH}G;9CA?#)umgN"
    "J4&$#wfCZPXhWS^;$94|_KByU7O^z^a$ypRVKqSFSxiV1DF%})j*qTKLIAX@vI;4KK~~TYQu<@`dR?QXRtmFZZ&4a|J&U)R6uRI4"
    "ahDeQh(-"
    "~z>g;$dI5vHZ_C@k!+Y|55gR*5FMCMJ4eP^&t3_QoK>=`fdnbmq<K>#iB`zP$91qShWnjY<_>=P?~QGmTp!(Ve*gS;Xwr52$4odi"
    "B3UHViT{Ay}wMua1ruKK=(bP+L{o1Z@SBwS@EzGtqqG=6pIL1%fAg;}f<8{y@HKC0;3x+N0DNRi2&I{5TABSN;Ky|sY3hV)5y0*A"
    "MeBVc+RK|OTANBR~NDP)v_s<KAm{g6Ns-8}QOFk$5-"
    "@otog<B=wiM5q|~??F=DvhL#E+SkvUt`|%!;=lN62FGvNo8uC{`y8X9wK;f?A?vWw3k^qhHh8IbV2Pm$n_Mz)Yth99Jqf5>)5+A$"
    "r71p-%$x`+%V2K7U<g|C-nt5Zr-RGZrnPmjzF#GJpebu7vfh4ivwA|o{Y$n@gsAFw>qK>HX*M>Ogr!L@`1eG2r@P<dGw;CN|J&HT"
    "?$~i1Sss5Eepb8le$lHm41$(QHpkQjq-"
    "<lx@Vig`R>VG3Ps_jvFf5jeRp*?{+&gx}TK|=@KLPi{By7}UkysL)H1gL#3yzp=?9wAySNgF4w4I^3&7jUksBP4P+o<%DE@AE2Qs"
    "-YobRPs=6iWMuq}sGIz3Yt5j*+Zsl0%pP-Z47BPXFq-qZBa7PBDjZBALe}_aX{o`jkcucGDxcV({Ew<4&#>?+=#6Jh|K%l!TYE_{"
    "yoJvc*amX)N;M$OGF%48p`Zsw6N437BvsNAomgB4L|o=q)1&&oq>O_V(#quTX#_I4{ZTGJ&l*j?E~;X}+#jQLah$;~Mh5ujZAk0p"
    "2^MPmS3Kqap+0vCjYqTL>({qIu6DusxCz6lMC>uFEy>STSNzgtg>*$_K=-"
    ";i1?%*y+$SRp**WR$s}3b44q54XvCO@vNi}UV@T16e=v1^p!{heFr)UpWzC~@@&-Av$tAgXc33=-"
    "6tX2Kz7zBBt4@td6;@^hplyX@Ut8cDrWuYWfzu%E-%H`u(g~iWDEwMPp=nE5@z*_(3kBRgutZcQP-"
    "V`(cZgiAvFW(b&_t6gQHMxH<G2ZX*gJ?32dRw9~qGx9=v5+R0G|%$~@Xjz;2kkB1n@_EOv4TtJ9Naz;s1s(Mc{S$r+b8&^rr`x;{"
    "E$`h?4!Sit{tPXG;EYn>^Ws)Q+eTIP|uhff(Am|;L2NUXQ_f#wAQybM<vgW44s_dbn?6cYiONrbAo(?{9L;h+q?GAv0WEUjAgR|J"
    ";LK;-L+uwumMJsXtS8Nqr$;$7|>XGlrr^K-"
    "vKAhh&*uO#=3hmY9jo~rtI5ma$7xTYM!J0EC58H!{r|FMe}4M`b@b5d28eno72oFs9eb%bX09wB=>=2WldO3Lz1q;<YvT>z&r^CX"
    "d-pgd=KY>**hq~{R(82|<>r5&)R3kLlRKmv|>EfRA?#STQW4s#!ukUXDQF&g_sxn!p>g1HCg$iZQc)F0ZaGnuy#_N)rq+0$b!0PD"
    "P2#np8BnYzz%aK+ZNZQqQOIJVNVL}*k=3`=HHPqokB*JD}qw+QCVV25hAHbDdat)+l#vVCM-"
    "@&b1s46JoaMu$?RHr&gIMf*Wldhg6Eg<9aH`ohiLHgbysnLE4LEn0^ZvD=<|ubcRHP|7I4s_B73jvGO<-"
    "8=1kUD7<C?iprBr(Fv^nd0^v1bL3=CBPFVpBj?HJ0eFU?1XM!Kq1QJ49>Bgb%&XO-"
    "o>39iax|gF3Lg^>{<0ls>kQ{V;Aj?bdp&tXr9~o?f{8tNmV^3qGlPvRw^tizVeGUyq**Si1<odw9lSFJ0Qv|D$TmwdRUI*#C0Q1("
    "=6GmS*n>dCbK3n)KzA^BdF_+>{zE<DL+_xaDr4fbB)cP&u?;=Fbugpd&B8rOTIJq?7hSo<xGN)L0UI21a{uei7KqE?bl}zrF_TAy"
    "Ph97G63fiWAwxljG1BxErG^+@ZfY4D-"
    "(LajLv3zgQmXdg(;7O^@7W$?^5ItcXRmj$34Ap3I;$qsa}E{UwQ2`s4VZ%j5J1)JW^6QXU<pvm%MJ7M*3aHNO~|-W-"
    "OigCwkOp{k@`Svoww7Y?!o5@=f3(s%hW^dy~XgBxmC~5*I)pK>aayz}!tZal#J0+J|}^Th_@Nm8r%-"
    ";ZBJ7y{;DlfhQscAD6}_U7BXR(>a~fsq>&D@GZ=CmLq_GMjI}vJ)nZ`L5^e()^=6X6ziH6;0oDojzPRB>Irj74!5m&#0Ro4GiWR?"
    "uoXq2P{NpC*YQ}yFp?&Xp<pdL63@cmAXKl{II;X}5d%nWY-"
    "@FTR&yv8*D*S~bFeZRAA^&`hw9CqF@{~o0Dz!UY>AN{x=GxHB{AC$ch!xQ^g)b4J+U-JNXeW`FJ`-"
    "5s7ZolA<i*;+|#R=aZsH(n7)?tYw&srDp20$z_mySls~faSC>oP7h>R&@Njm02bD}f?ku@49mAAVA?SKdvwlHE8Yg>6rpwa6)RP5"
    "Y2++}3tT`9jvzk@1rDta%N^E*BW;SsvW}KqZ=UJX!fyf&KrbB&S?RX+{)ZL~w6EmgxjMk2TuQ8AMri7Fc`>g}W6h5%l#N08WIF5<"
    "R0dO01pTmwpJAn?-wdb7-"
    "FSSglz1=|xqu@v|!Sd2B!l!xa(lTqaw%f+SHThOy86&?gs)jh9<@GfQ#enr$fg`N#gCc$Ll=__&ZFDnp2OFF~d#>%&dg>LMIG+hZ"
    "Q;izUw$ugW2?M@<No|$tDB*a3BSNPH1HHpUX%{Ho3<uY90wn>y8?S{c0|&9Y+ehm9X|TCO9Y0bhD&u<%Y~ddGs3=LbzY|%s3}ARR"
    "3Io2^LbQwSbD)S9?jAs@S!Y`R)5*FgCh(S#y{8~J+yBmFA{RlT%Q2!1SwxVdo^$ma0bsZ25?YgIuvKx&=VZ$SSl$4}0OGlI?hst7"
    "0Y&(GWw~7DZmOpfOUflN@*M4ILtL_)MJ0JPCC@CzaWOWm9^;u0Y+TEC%`8IkAXb}SdYZ<p?=GX8Hm$P=<(TOcBTBI|OveF|8m)!k"
    "Jp$~pAyqIi`L)0A`Fz2-?tPT7D-o6WW9$47fKNGfhNiz9ZQo8Pn#Oxx#n;e1eXEt*dmAhtrWs>xew~ucfX#=r2{Cofzv-xGSLcvW"
    "BsWkv)eCja+;Ip=REu$ByBeVBJ|+dAPS3QurDk(dUWoK7KQYJk&>iRGK%7}AH;GAr$NPYM1T=l5u$v;|uPwqD<|qYZ&*B}NmbvnS"
    "k0GG(A04|m_L2+J9YG4Igx*O&UV4kbuOiWl0IhqZ%1{cpuabF%SOO|H-"
    "@CthoJedjw++VUWOYpxmmbNz4KSsOconr?1dq}Byuv}B6R6RUb)f?@^bCVev81QqcI_vT1(&dKN(hlbTfFePY@;u(0i~Q{5NU5cT"
    "+5Q*QnA4$+h=!QllXk}`>e=RTO@$q-tpjk9r6AnDGa%TV)yDry=0?vaIg+X-"
    "fpjH$$2yVO{c%dDs0Nzpf?OuoxOl`Zli`D729)ruE`4$sx^d-4B8t4gwb(rj(Bu)@TMxklIN-Xlt@0{t@j9s6<8D1>gbAvV90p63"
    "9xP<>q1#)PHX}&XwYkd;aDX~s@CRj>*EdzY!cfcklI@2vjIJgd}OVRBZpvphW)kdFGphT_Sg-"
    "+83Xm;lgs{AXGbQilm6SWr#I&%u~h&8`#x}6ihg#S@PIl9%dmyVW(%(2l}BNS=ER$X(xF$}Gl)v;F7H>X%2%OY)cU{=-"
    "F}{8IZ0;W*&+&ZKhoMIzh)lUOTyY?EtC~g2gnoIfYxLW$Ey~cGo6BaDl<}xQI&OyG{cY%TvqmLdK(DY%ZK%E$!M8Hznuz>%tRMtu"
    "vX9Un2QDR;A}FPnxJhiawIb^c|3#5dQ^E=8qUs+m({njM!v@Jl7-"
    "i~ym0J;HNjSQN8u*7V^Gcmxb!nmi55LJRZEc~o=*S0*#_fllQKYWY!4s34uv9w^{~U2>{>>K>(s-"
    "1RWZ+^Gax^Hjc~SPtKnTan)NsO2q=`6HBj>MKV;x=h7EVFFMQY>VhdgL)Q-;Qp|uI&uk`qb(jvqRn~Eo+!-7b7<QY15zv{bm>-"
    "jD;3Xfhem&7au44!z8Y7g9z%sW&~+c|7`Jnl}#XB^TESciwEoQBJrYQ5y>w#}STfK(eHL50jVx)_zwG$+<Low0R(k;^duJ*E!w5w"
    "qqPQ}Y!H=d%ZN<OiOSOV9Wk&j~Gx41feYbqUH<qVvoRcUA?P^)TrkJS@kMK;Z1G@GaLE4wD__RodCBw<tt<kL_buLUq<njwS^cSz"
    "lSHw!F}cn=<fvMwXxpo_A>M)*7pc<Uq?{KUyk$f>IIUZWfQCGtg~LF#-"
    "`4btj(liWZc+U)s)yeqQGwf0q}f9)jo!5lhdq){D~=ehwy8l4uiR2za*eu3DMM04Yy-s^sg}t6oF{t@;1`Y9zM-HZ+Z?UB?q4(s|"
    "g#i}lPlh(b3ID$WaQgba;Wg9CB3W-w5qJ5f)DL{}phT;rS8)iaw5wj`nx7~UuLnqCmUnFF7Z6%Vtg=xp;ee37fAzbUKEEpDly^-"
    "8ZwwRa^ubQ->hJ7JFml+T@Co$=)eh~s9)?%u7XE4>{^mFXG-k+@%ICQDRX5vGB%-"
    "^~ascc!t)YP7UTkn8jV*H9g8$FIr(rSJr?u}-"
    "$q=D8=A;3E=9S#`AhEbcF3c8&A$X~h_qanan{yjXHA;vEF|9AI<=bp3Iw6RJQT0Q|<2<Zw&Zv&E9qX9z;!gnI=!)Z01T<sPn<^N|"
    "`?!?*?9>;*}Ta_GF9j0E9wC9aZ)w%;{xwZRJUX5Zi>w-"
    "m!p6~}A6U$PHttuAw4)iYiIyz)EY6uofPRvaMe^SwRRvE(Ixg72r1sY2JhVt~bSSi9kI753<O`xv0&-"
    "0e_BbNX3h7s(5?)tpza?=lXupCbI8-"
    "DNV>p@TzuC7WvWe8h%<t$XsTPToUVkE2$1vkuF8DrZ2b+%Ua%&S~0_IJ2Kzb#EMjhCa{knmpv;&WvOz&f4SrYRK26*drEg1CI0XQ"
    "*kBjQo!Jl!#V6vIr?vVt1{AIx{P<uev#xIDNtDu*KMQ(+eRpn>~WnR2EdiAvp=6X<87z$tjDd5=fJormX`r<%arC|<s`7{WRs(=_"
    "Z`4e2{EbvdB|a!ly<KmG>3jbo);)xP^uF6OjtmVL>*nUa5S}=hJt2-"
    "PDeSbInbWY44gRHeT;oisC5qJIbw+}v}VX_4BTBdMZ30{K@+IVjy8gIBs}stNWhrud6*+rV{Cy=eg&>sBt%uohhOSF&gso1q~J!g"
    "Z9_qhpX4bX(CSjDdak6EWVkm}56~k?Pd%@ZM&ZbIaO6034y)A(`hx9p;H`IWV2)TW&Nut@zKl-"
    "2JQgC&X_)a%?zmhW>CW1$Bv@|(NGqsn7|!+7&4s9ddgPqeGaGvhi04Smc-Fjhq}7t+aQQsiNe6R69_R5T4Q#+&FZXJtPa-"
    "E}IYo}%7uZv)DNj^llN-2$1eij-L?c@t$dw(GtAR%9CFQu~jc~<6o0W~k2_#g0)awBgMM~QKyT@(HfY-ygfnQw1M!F?4Z;6Gx6^s"
    "V|ytYnt9Kn66QMBVS%%r9-"
    "v7u*8{b;=$>6tP=lTT>>nX6+d8^?{c$k4oosho#{V$!%|>7rfYeHI)2yP=+ogiRvs^GT0Ujug?r#O=X7!rX&=v)C?QL!R)S_5`o#"
    "?({?ek5UKjJYTYOn)Wq5F~GWe3_k2JTXK!|d&Kw#e@)cH#nTmkAeU@&rgeRDiW1#I0#!lIg9(1dwrvtLgj`5CI2Kf+*Z9y=tm(w9"
    "ribe>a7KqXQYTEU#omYKjg~6ptD`k+$;D3Z<IGdthniWnx<~u5HX<XL#6lwHku9gEh}gS2<?SsyzLO28S9k8OG<l^Tr;F!Q?n-"
    "r163n?VMmj#0K$JFA%@UQuxGi<SDgQMewOIyGIpAN!jcw@pWW2do?`qu*(+dx^MwENKunXi$98=j$%c`NSqd3}v2)?`cko<G=A<U"
    "FjUDGP;JPJGpV~V|GlEj+3chsnq68O_2fbwCsf0nCMLDc3$Uw_lj24yFzt3};mCS;kIx1~3!Ij5JH8W2`$;4RuIsYSuSjj>_;_sY"
    "E$-m!s)j3t0<fDH2>nQ{m1!KY~vp9A?n=dg|6^O6FRIAoA`-;LDv+DSLo3?l=9k49AcGi=A06YFpg#;|@h(Zis~2-"
    "brdH8rhIgDXI41VVTW&bflK(H}yJG?411;_;@tVS($w$}IcO!)~O}Pp3-"
    "=X>HD_XEqc^%bHA3Ht~5LtyA#F1nH24cbbFqyl*Bl8!cpX7uXJB^}X59n-{&Z55wqICm<p7%oDuSqtO{nwlkHGmq8Q=Cl~!U&-"
    "YM|6cj$hspnfe9^%XtpO^v7sL@O8(JljxWjBu&9+YHLPG3O`vntNHa!KV_HQ+7OV^3>ZV8Yy+p<d@6O5IRIQlhdc2o__xZO&<QS7"
    "CRI8H)XUGbq|Lj++h5W6i#|3VA-"
    "q9Q+Q3OkZWRF)}dFca9Tp8n|e2BP<p{;WW6|aX&~th`D;Up`vf+Gv~Lqlo6+gf2dzpE0r!KPjuqo=jN2^1;%pBUC~E(Cu_+lx4}m"
    "50~6=2wZrzO09zaPuq27gE!?4Znyw-IvF0sfp`r<N&Sy&mGU`4LFB*GsYO+2E=ozRXp*fFUjorOxASVYEZ;zPv_}#SSmlq8@7_Yc"
    "kt`0_t-"
    "H$xV)bHB*Xpj@T^W2^U*eb6R_eRg^jF!*#iZFUfCe};pmK1}t!khWe2B>0>q@3<>o?AH>{6t##B8Zp9OVOR^k_VbpCt{l&CSHVnZ"
    "PuF;J0RV-!55l&Q^brk4nuN~_%B;h2%!C5s`HqGz|^4-"
    "vvq+J{yD(9v93`=C0te;m~yTvXA1E<RMW)S_2Kf=(X#1=Ggqs&k)z4O9R61SNJn4oS)8VQWUk%#(JH_k2+FTV8wDcse7fBvihNol"
    "hZ@i=)EX5ODchxf`N&h#Y%%p@*em6!QLk1ns>J@1pWS2+^rVJ=-"
    "2!taTdl+l!*?U+(ud#!dtQ3=RE|`a&vJXVjHaTH^>HHBjLyD3iEx-RtJYM%Mjl%E%N=Y-"
    "OmI@JpPS0#zD!>Sv*sKW1Y)^ImZtUWU^zSQZ5TtPIT9OKUphj^o@kEcBw`7#dHNkF2{)~SWXZEmm?f18Cc(h28<UE%r8r(jKGKCu"
    "WJ%<x9S>(<7_4W24gDa_@mF(#1uWG^#3Pg*<#kxo?Wc|Vbl);JtLm0C?!bN(Dbxcu9Z32rQVq1+K0=7ZOT7YoHavz(#z^vFCSS(E"
    "QHwOu(AU{B-RzOkej6gzuOUr%wm^-KtMBNT;w(P?caJ_qbpH;xiMi=j8@6Nx$U9}ARs(mfm=?p&u1*1B!wHncwT-wC0~OoMO5?YV"
    "k!My@Vo*}r9+*P)=^9>NOV&A-cJ5{&a($m~!FisLUR$@?rpj=p7-"
    "vQ}UM*J>fjQH4W`QQ97=c;Q4Yx*EUOGRvurr{dQgrOeYoiM?$<NSKO}!^V$|df#!O`I&-E<pl;~<!XL}B7krE7CI?-"
    "<T3pZgFrFLk$uwSBn%tN#a?sp-2LT67|#$DQBuCXVXJ0>4IAnnz*8k`e691qR)-"
    "#kEOyjePA&dvEWHmY`<mIH?c)kj@FmGNVt>Y|Y&7gr7Y(ze#j8V!f|qtg@~5eRZYrOjF?5sfkm9z@ZKsJU&M5>zZAIG{2m2UakHk"
    "fcx5=MuI#^n7VL6KHF;T@O!#I&_;uNdX!E7ip{kmAwcdfyDC+G!$R)3^A*!AJK}k{Je|q`b$bm~!EENF2@jXvoj#`UnJsk6K7y^0"
    "5r?3F<2~Acu+o`GT>T@r0@c`l=j686&m^kcv;5)*mVR>9Enig@4gf7YJLA#@E*LzrFqXKk&cr?R9LJ|zrV3=!b+|mD-"
    "7#4*bb8$@dQxG~MQbx~nhqjbvn;m#kcg;^oQ(###wii{ZHfOqTPwN6+!8o7bKPL;8%%{C7j#jHZKp)6Z2K5I2e>wu?`z3@(eTPkp"
    "UIF`H@%KBM$a0c>=9e1nV3*TG&Tny)Xg#eLUe1;#hzUy1<1OhUD3B=;wNxm*sQy1SX6D)dGP3L)dPQHgyh@y&g$H*v}2^W=wtM(b"
    "B%+=)QvT6Sk0ICXH>WYcwwszzW55$Veo|<=LxA6eVJoOXleY{*cawkgNOnibvY-PW#qu*{AA{~W`H&V$pLZ0T7I=ZVlS3O1ql11_"
    "9gAxPAwIv4_#{pXIp>?*Gtw*{VGp8_}v#&gqm2}z@fC)I=AU_{K%6chi;_P&4lNBn8M54n6d=r>3CKOV~Z>Th0tzw25*(6W<kQ_&"
    "ac}gUIEMNk~!Ad{A8)yV1$~4qhmZJbBp#l;QT7XB<XrjT@I%quDeeBAeZ!OWW-@~b7-"
    "`dGq|{vax4$FnMNZETIQa<9COz3*PWO><6Gbv=it*gC*3&XhJ!8A4zv>LDfr-fbYRW^sZ}O>Dy>?U6Xz`w?Vexe$&I4YaS;pbTwX"
    "pnbpRClJd{Q=J_Ar52h`e?0Tg>;)Nh4MuW;~67&=b*&+Dx<nD&_1an|g*S}ImO1F=nXT6JjDlsls71X(B*rD*Xhv<Lm!R$%%`9CJ"
    "~Q4|IA&B5HIG>_|3WGuI=++=f#l`ghMC2>n3yu--03c=ufQU}v+6IlRXm-DfuxG#?vg+FGNTDKQP<ka!hyPg^X_GRshbYD{H5aEA"
    "!1^En3Rv}Z`3BeJfLSoTt?LDI5LpT5qV42)3D4ldErmLK)(J<~Eck=D~4vdm2VJ{r8G_z7Yw2eG!DL>Flw%C1)j$tk*{N<VMOc*t"
    "hK&zlSN<id1@)e&i`2vX3;(p!K6?$HxWDlrw0X7V<40oacwLaA$;h;LTn=$6po>N~M{>wnnTpHH@}jpixjY_Ir7w<6N8(Zf1GxSq"
    "d^7b3sehJA9`!gC_n8H1#w9F}yO^Wf1tgVPoGbeE1PIFo%s_Iw8uSPe!y;OWe_ZyoNsg*A8yIgdg=f7iyOEKkNq;jDR-"
    "z>2J0&DOSUg2%b_!b9!zPy)Jw>)PfKUV8?X5E=Z^p;3X^^YPNSF0GE-sG9WT0Qs(N%I@G<XMmM=H;JRNk3WtCYww#@7ucq@+vOVw"
    "5o$V3xCxAO@_l6a>0Z$X?S&cV!=%DEk~KMfjYCaT2@p{dILdTpcPxb*UQaOtbs3%ylRIebXVrl};=SnT_O|3GnY|>qk^w%UZ#1Pl"
    "Vs-DV;;V!lAj<0qBu#&jU^YUgUdeAQ7aZ0Ak6Jkf#VJ$FuF;MCIoeI<Z!OPGkehQE7C<}eMe~k~J(PaQd>hn13Er@~OJpOxeS0=F"
    "V$Cy$nj(;S7lVGM;~{{Wv`=WSnz2Vc_f)aE8UZ%-"
    "_&}2;P$^fV9(9Kix+Yz{tLK($DWCShY_^6e!@(yWa%ST`JCD(D*s?<HYKGhsClZUhj1MJVl_C>LyB1G-"
    "6g8lPp;_=j!A2?KQh?ATylYI`B0zE$yC^ifaBvgpIVqo}ZKP_iNnWrZ*?79?Zcj1^y*jW7IADP>a8ABjPGi1+KyW2at;c^c^FHT8"
    "af{PeBr!aX%RnzNI#kvWA5ozZTS<QfT%sv$e$s5-"
    "B|&w<v+fs6WhDtt^DZE<mTk`&TG}BTYdYml4S`ioAyn(l6ing$8WNV~!*C@`9tov`qqfATvO9ikElu~AwexDPoCoXZj+-"
    "d~*;}(QGo+&iDvZO><v)wd6v<b{R^w=q3P)<X4WkYMNt3e}9|<*Tj2E^8tBxbVNP)W3NWY2(JeMy){w6Gor}1u!j+!<7;6XO8U#)"
    "SVL%szhTc^G$Xn0T2Xy4dn*v|api&vfK$+?b5i6E&vWmHzTEr?^WTQx=+EW?hr15`F8j!w5nXLo(gmWb?UXD{N7S}e26Nev4Tx(|"
    "p>3Tw#crM}VO(ciHz!a8&SXGT8O_cUD_0|>&AP0`S_&q;==clbH3Y&8usMfn=8I%q|>%<?LY9Jg`|IUh3|pZ72;#|5D#c~+izh(K"
    "(0rLDA?+%NW_{;kS%F4(W~j%vfv79^obd|S1#%yR~P;3-"
    "J9_?aVmFXyBBaSsoq)t@zVR)L$6nceo|Zc|dArM4KWpXtyJz{}3C_Ao<150^<)HF#~LmUqH(*0aI=Y`}@Jc1j#(HA&;@Qd`~JJD4"
    "ssK%!dp*_wvtT(IM9=9flGGc26g0^VACzd<!Swcd2YuGfSX8q~1V4b>S}p@vHDAZHq={pGwu4?hC$Z^c@8s88iUL9>lM^OemV8*c"
    "AH#D!BOhaui+Dbc!Wz=uH6lx#f~p$tUEIXaYBH?_@~YIp1CS2-uyk~E&m_exMQ`6RC>W_uTS{%FaxgkLRJl-ipMbrFfL4bx66o#C"
    ">rvj&K`gOjRcj`*&0*mFV6wUS;XyD&IUjK^qvZ40kh4;|vP?IzUqDyJjv_bO@Si{h-"
    "P)A~B=TSToa+OI=6Lr(+OGBTMuZbJ9_NRr)p5BH@0ae!qU1mk3{R^<Ll)W$1MTGmVe62X4ADsp!su_JR)T>WMa0TEszwonahJ@W}"
    "%K619Gje=1@S#NrH4Cg1}<r(jW{jZX|adk8*H43J|>OuCp_qdp*&as-xa6&8)FaJ;>9vI;fWKh+N{;B;$4{2(qfu^=m6Ddu^Kus-"
    "Xoh(ts-"
    "Q28nTIb@vu__O}0aIUy!0Wk>iGsukDkA6QCSTu_2?8mfW64)TrSr;Fwyt@YcoQ=aDm3=V)gT;`Dr>bpML+o*iSDX=OYwR&x<TKat"
    "eaG1Zp<3^!I8uyzS$9(COGPfgZ3U>N6lm_ROs|0DeS3HY?9}1d`wh6nj>O{o~K|WwrPyb$ufSn`LzR~5;6!auU{~f67)~-"
    ">sN5oy}x?h)M_#Vy44z`RDuBIAN4RckZJkaH=RhwbzDgbB8}Uq7rq%w+Ic1X-"
    "9)94$t%V3E|~XP%f#qnv8ybph(?a!T9F7ErtX4fNSS19nt8|3gQc7+B+oJI5$9Ezb!4Ex#l0rasgKKXWJWN$54BE^>W-"
    "Dqng|D13y#yn6rjHJ<*|^QX#{~U3|R$F;F&G#?2L&iz4mlyl9t_IaqF<tQOJ^g1Lf3`cf<U44vqVo2B6iyZlTNv)F5UZJjmmie=V"
    "Wkf{64j0Qj*av1tV|xEqWih+VfyKF=_BgJYJ)1+}HtS#y)nlA+?4ml=5~{ifQlU6%T^Cus5>VQ8Pj?{s|=sx~+uJHhAWFurNBy<)"
    "dL&dEn1lWGat731g0c-"
    "^koXq;^DF|HsQ^r|BlcR3SKFzx3stq|;=38X#Bm6&vRF#%OMcHw6l%bXS5*WTM$N_Yx%=G@kmdAiBA5<gl;=h$D!vg5Gq#MLyF#l"
    "o2>j#d+*KLqPxZ-vb$eT;=^BeH=x88P-|om_Hz&h+hRC$K-GnoiAD1))-"
    "lb>a?oqat}oDt3omFpcg3&shnvWKEE#L0s0x(;+aX?AqauA0KZoOz$p6gMWABV~AP@mhxT-KKq<sYurkbbY{tgKW^)i39;5lLJ-"
    "%f-6$5fiK9sYKW6jf@pNT;VAU%zLpqM!Trqw1Zb*$Slk<7P2M)ui0>Ht7$u-"
    ";a`ezHJSTd;C2iPfcBsL&jp0&{ErUl6~aWWXWNC_a&g_)8i(w7I@<0<q_oJLpnwwdUQ-1#u-C+4(jl(fgf-"
    "&cH1=XOfzT8H$|0T?OJeD7nNvn}C?ol6IM1e1%l14~X!YoEUTUdA@r&+97;EhBAhdIyes?<N5%-"
    ";rWmL$yx83gtC(?x`wbBOu35vn*ciyVVaE=5u`3sH5#+b21~gNLzUAo{(1rvzM*AN1+bTstGgBwVU=H6Jl<+(pCT}GDO5kMdLuq8"
    "tBJXYmlku)+hfZKt@dl7ixg`L!dGCOm;+HAgjMS{e;<VjQ^SAf4a5nHH?Sg@gZ3*%iynYVn;-"
    "&eLaGRSw(eCC7=(^{1rt#<p`L_HQ3fhG;k%V{yM^}af;tUm#z-rIghxRIWyJ%-A2bkOlsnsK4hs!BiwI{M$anN0HV-"
    "J@nE_a%72x_8#BD=<-`G;Vz-nwI+9K1vwp5K)ErQ8DOQ|zR7H$%QxB@GI>tV$&8^^KFoGHr+QR)Z3&*Y_ItTq#(hU{S7Ei@R-"
    "&sY(74AI%#j-"
    "UT2kJ<p62Y^+Pl*+r6j;Q4HOZboI$zMXx~9JUZ1_{1{=OzaL9xxTk6fJ_Z)m|8qFLk=)Fi7tYR5_^?idK$uY1s9pp<Oq6wABoOE0"
    "#JtV8h(JCqQaY3DmQ95_Jzrf~|!Lr+1LfJl4}mG*__UQs?>hq7O$Qr0UQy=u|uzBT36SKQZ<5IYOX)rM5Ae)qKXNGGL;h=~A~qar"
    "18zz2n!A8Z|$MX5_`0bnlN!2#Fti&G78G3FwOs&j^^rw~k1^0gN5L+!Ix1T}fqNKWUD6q`=FHmLbiMkdA@dsp!Ep!w{KD{MlIyrf"
    "Yul})Vcuvi=Ct!HeQcU5j{Mz|a~_y%JDen#?k2Qx|o*OEIOMZ*Tp{JOcE9OwKgyZ6~)lxLhg6JIG^;)upP!`J|?81@p;*H<DW^@W"
    "yoB=#CbGZAouf2BObtgUf>FUQ)+UxraD<jT_4d?P}86=etZZZo5_bjdXIc&gxb7`D2snWhm8)H8V|F@Q6VNM-"
    "I#_Q=e7CIHx)2ZEVr0oG{Q2Oq4)cO*0S+Mn^EYj!O`zi$O4Wu?%d(}7&)8Z)rc(02-"
    "gbu8%NG=1ihn85w&Fo$5jEWoL6=cP^6Xau_LPFFn`v#KNgDh2O~N&q?wDdM=2t5NMajL3!lXWORNRY>n*iv|7|2SzkAtt!VubF1O"
    "V-g2(EX-uZ8>p|TD<5qf+S1=OtpQVOCXC7_}6-"
    "R_^M<UL;t9tH<0hFmYZ_5aqW?WKbZC%h!aRAI|@6q)RkWQj9*sYN4yF(4lK^Cd=xO5v*u`ID0jkwamv%W>y9L8`TAQJw9Z=1TPdl"
    "pMnFCq!&Lcm9<K_N9)nReE2<#LWUSKTsjZZoY$6?L{EaVXEP@KA#K?=mSEe|wR$Tic37M&y)%0Dli!c7RkYJte0b-"
    "rg;?k|~1ZuAPoK#<VNA%C5ZjDRay-8|O9--g3_|05p>Bo@D~0Bn4WC==m}k5JBBZZ@pd<{yX7-bCL-"
    "<c6Cl?lbd!LW`v{?#8F8b*E9o?ex}WU(dHph*CpFi_b{tX?R-"
    "?9IgY6gMo!$4=v;>Nqj<+I9t}kTQjZ#&JsZX$+ljQssJtIzKm3m4!g!~_dQ_SM@_-"
    "Yc6=t+7<!w;v6Fm+KeCe{`Q=G?}%9>QLeT&h@Q&o+*XM>({>NiIn8o#7lke;L!dNf-G1evujlG)MJ5d(bf@iBG5xhnc&k+a;_-"
    "R4QihQEd)Rog_QK}#yjt0uIr9_6S8<Iy4@|Nbl*@?f`@T@-"
    "01*l0Cuk(uN>k)_ZNcQ|(j|9AA6_S1&W*()tc+_zKBCaxtSf2pR+fCuXnok1WK1-"
    "JTQd!nwcD>AiFM~Tz3T>~LCl~I7(4urIZcs+HTdeVOBBCR+>WjV)NA;oxUwU>vFs`dG29>iK_4>(kd+4GVbB6ACMBBQ>RVP;@um{"
    "@>JX@@AxpR{?|DfW>P$ltvMeL~ATG7d0FG%gh>`!Tqax<jO4@fNqVPwv2TH>Xmld4~@OfWwF}<xg)_Mmf@{_N$FJ%)_!t6_K4kh4"
    "PzRW6ha%hUIx&GtAtxy!2850mFV5-`LhA9NxvAG0Z-sFIlOM(5CmaV@L7~V})&pTxhtlxLc$O(W3`Z-"
    "9op$HnQKaxTxcnfPP35@7KgmI^K~P>UED4WBQzc>ZWDTjuNotx<+JJHkl*DOXX^y80QFzTS5gP1|3(DVJ+_3uW6&M1cZX&@-"
    "{}2yx}Nx2uRBaB2Sw*{UT*cuPl`S1kSOcN!FcC;~3Mh=1cch&CxPk#d(=(-B0URm2RjxZ1J#y!n39vFz&t4Pa3Ni-"
    "w@D!0j4+^ldbq%hiNF{r1H{>&xryVL;fs1s<mTK-"
    "}JxExKA=sY3v_F=84#dSa$sJkKere)4Ol~t7nxJREPX}t|LIk5yrJon$XfQg1bAp0HyT=I^IMwW^oRxH`9RlXXwc1GCUbWEOV+9J"
    "LEA`FINKSbZ+h;W^sr~&^>2ied@}E%{j=>uJzHuNg69h!Zp$Yr4kY(;GI<T$FngIPpVc}eY+MX8VTvo#_{4eI)A8P645g}%Tt($0"
    "C!vW1e}1;8t&o@*INdcV}Dwb{#wudmAsTA1NdtMtC-d;-"
    "N&7A@A^tAY?uaxq#Wb44lQi)an{`Cz6TRbZ!t*dX&o}}J9yeC7V8XS5Ss{XD57&<Rdsjt$>R$e6gz_3HGXg@O^p*a!(yNZdF^QhC"
    "*zD<btl@VXSot7GiBS30eP1sa~29|e$Czy&K!%@1UHq~j}~-EN+$-"
    "?_Q!vH^Urtxc<#$YFZ;2lqQFHcwMk_0s0~Kn>sWYdztwTm?W{mD+sbBqLoS_enWnI$V*Ah9A11PRk|G!%x5@+y#8Y2G_hzqs27`!"
    "CUli1xzZxdY62mTY<()6-S~Dep%FI2(zGE=C<KJ$r2IZdZp-"
    ">F>)xHa9;rbOy>0B*I8hDPz@4;vFjCV~7@y@BDyM8n=#7L4^%Acs3p|i95IR7O{-"
    "GM7)Oud~#QTpU!lTPl!RCgX#O+p8to1eGdQ@(|bHb}Zs(5BAYGR1++;9B)tq<V*FV#7VMCaj*lJ0{IIT1Uzj_5=G0NL0I9pVg8_G"
    ";V`3-"
    "$9t<;WvKjgOuZ3MdhX`)qoN$Cs8MK)MsPI^yOI9NRLKJs#t>OHG94d%d&8$8<VK__&D~lMivx+VB0X%LHqjSzn=g0!`p{n|MvOq{"
    "f}=Se#*D`b@k}o_0M^Bar1A#xKH&@Zo`2?u7A?=o?Bm|9!@x0X6tp|$A39@JZ~er<r{4ot?3z7gCu*m`P3Sv!&YCjMkdxg(7k*OE"
    "PP3_Q<l&AS~en!L&Tl=nhr9e4)uQRYzNW<4^q^})eJt4G-"
    "NmwpNG2!ju|dHoNZ;1sITSRoU9p*zkbQW5docx+LW)Yajw!mY1h=tOFuKkU2IADTJ&5Ve20=UJ}3rxUag5G)(___mIg}oc`Zz#;N"
    "PPAdEeD40QqBvV9~F?x<?p7Im#tn{qXol#pKCyUYfb2Yj-EudoF(g9s|6@9Q9!hH7Fm9q!|m(hGrqLNqyunsfcc;yC9|ah;IBv$~"
    "hPt^CAu5dg`S*O+la*A}CLNh-"
    "*3d^!HJ+OaBY9XshuPG9JtoDax8RA6@B`QpD0Xk5m%10gXv>E}e?qV$d~P#hz=yt_~AgBzq;ZLt>jIL3ZBVEy5#E7VQz*Tq7-s-"
    "ic|#p#JE`%FU#+V=l4XQHfDM04H+GUVY|K=l}&>86$7QP;Of`VKSNMme>_sSJEDPw#E8$#W1C()v*P1A}q97xW-"
    "}BVBZ^~JDSsYvsv;=$4$^ximCy0&S~PV$NMx;O2911Xa6)SmhDok;JUV%*CHJsm!H;|^M~&XUgqjr9F{$(I^*$mh^`hT5Btd6-"
    "7V#y%8wOXt7jP8=)sa-vJ`zb(Y<$~n2&Qb&^UE9mT8xoc*QjDA{?3FR900LCLoU%uF89aN4P`zz6NU$r(45L=1ta_=SxgwmaUC)J"
    "C~NXWlM0V)RoS>41Sk5U~R;KlO*o1gMhbP%w#6BCGPDrjAS>EK3EFUJsbBo|7K17e;2YsBW9Uqk=pvh{GE&Qdsp%IE*OxmO~BZ}0"
    "5lKxnmLq*UW>(-"
    "G=QJ&UB|&0ZTroJ^X1p?v)>qOo=2?MVldWke)mgpt}|FMxpSrbo3Ab5>@Rt4C_KH<H9HaOI0Ix}*BMxn)mrVAt0yKftP_sMeW@=c"
    "7a=F2JD=H^IJPUem%Uaw4_j2j8qQ}LS=PLoSS4sJe=%d(?W>{Bm(;FFVVXO3As70os7z=xmhiJqaW@bTr*M^e5ndKn+ca0o;wfg3"
    "?CYc=n(9eD<2@92u0qL|2<FYaRsDN?*TM&arV3A)-"
    "<UQU00@y~&T`M<B=E`M;aR=rbd{GRyPrBOio3h6w)V1poTVXOGBQj$fCskv(V;RsHyG2@OeDT@k`&kA$c-"
    "%lXWf%T<=|Y;#4#mQ<yjMiD*CqMBVFWh+u8677<r~S+SR*F5pxL1Yce<1Ow4J(Ne3oMUfYeA0oksr9@@l~eP=5*z-"
    "&rzm!6)!CE|Yl`I6&kPitq(B){wE4#A_zJ*OfrF>-=X<xH85u2~|s2wjD;@?6xMCHlT>r(SJ$%h>C^E!T%Dyw;L+ty)Q-"
    "$65n`6qzM2u0Gc9JWp8JIkSj4y89@QK1a~j%h};wMrml4L4?)UPPV`~#jX}YO?z0guZ+)1&RT>?ez4z;2WhVib({!xQkps#6Of1u"
    "BdX4I*rKLNWNpr#^(MSHb={Ug=)Eugl6v18t0-"
    "9?A_eVLJ)b%;Q1a0BFc$4wEeaaYQGCD98mML=!+9%h@PZ+jvPwoOHjhGVj>sF*V;UA@B&-"
    "M_xY7N%?5b8Hz=G}Sundp|VsLhn4$ue?i4VoA$}_%XcTFcCq2&2c<_JHXAN0MB+?*6RT#Ny==iY(YQ#BPrCnZ++FwVuqedX1P--"
    "$Mp$LzXaXaOY>Ywt3R(vj3J$vDY^%B`|yv`}JsqNRVcqW;g-Y)RVdT?gR17nlEeaenU_{@w-"
    "SRz!w66UtuIZ_Gk#CgnN*=F!1CPG*ZR&Gme?p<myA{N?R)wxW06&VQJ`dH3bR56{2;`uT5f-"
    "@N<r?bi?QzWwLB51;=0^XHGh{^iGa-@g0hm!ID~-"
    "}UzP>|0B;D(myNzW?~;*DudT_U`$<KY#q`!>8waK7IK0?fJhyfByL6Iz9jLkB{HKz4JRifB5**`~UC1@y)xxeE9smzxcz)w@=@Hd"
    "H=lN{Oxz&?)8h`|KsiBpa0^g{qyTjt9O&1SpV8Twf>Nw{O#gj^V5CxU*EiY&VJYPrN4jp^!|VRvbDbce9Ln+p1=0-"
    "`|sbJ6`$LKN$=gK&+opi6uWQUefR0(FaESVf9Hqy-"
    "~IgdgFihl_wA=&Uw^D`{nK~v|N80mwdV^ze13cTKX31U`T5hw@Ba7u=R*AS*LUCk@Zr;!Hy`__x9@&^PN~0s{`lQ<G5+-"
    "F<4@oF#6JA^<A>K<{?Si*R;hR2J-"
    "_aG{h#0d@w~?`ub<Li{_y?ZKm7FThd;l48Q1%_*Z$*AAAj?~zUzPc&(EL6EBo#*Z{Pj(m!CgA@94{S&n@)bU*7-e!<V-"
    "|{I4%R|Mc!_f%Esb&tE=1U-"
    "|ak$6vnxkDuQD_W8r7|LFht<v;#DKi~SNx1YZI%a0#E|MmSc{r7A5td4&$bYIk0%HVNDp9ka*G5n?ZtUP^P-"
    "{*193?wXau$pnTpWXit$)Pf{<pKNV-G{&b+1J-+&0EX`g~u+>eUQ)f{hU|x<-"
    "4Cf98U>qOWl_7TRk2hetQ4;gX?qo<o)aXywUSNXLs+fKmXz<j6eS4=g)4VW&c+{0&m~H|Ht!C^)vOafBjzqV^u@"
)
def _load_embedded():
    raw = zlib.decompress(base64.b85decode("".join(_EMBEDDED.split())))
    return json.loads(raw.decode("utf-8"))

_SPEC = _load_embedded()


def _rate(seq: Sequence[int]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _run_length(seq: Sequence[int]) -> int:
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x != last:
            break
        n += 1
    return n


def _alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < 2:
        return 0.0
    tail = seq[-window:]
    if len(tail) < 2:
        return 0.0
    return (sum(tail[i] != tail[i - 1] for i in range(1, len(tail)))
            / (len(tail) - 1))


def _exact_alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < window:
        return 0.0
    tail = seq[-window:]
    return float(all(tail[i] != tail[i - 1] for i in range(1, len(tail))))


def _payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def _last_moves(seq: Sequence[int], k: int) -> list[float]:
    """Most recent first, D=+1, C=-1, missing history=0."""
    out: list[float] = []
    n = len(seq)
    for i in range(k):
        j = n - 1 - i
        out.append(0.0 if j < 0 else (1.0 if seq[j] == D else -1.0))
    return out


def extract_features(mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Build the 85-dimensional state vector consumed by the MLP."""
    t = len(theirs)
    if len(mine) != t:
        raise ValueError("history lengths differ")

    f: list[float] = []

    # Time / endgame.
    f += [
        t / 99.0,
        (99 - t) / 99.0,
        1.0 if t == 0 else 0.0,
        1.0 if t >= 99 else 0.0,
        1.0 if t >= 95 else 0.0,
        1.0 if t >= 90 else 0.0,
    ]

    # Recent actions, newest first.
    f += _last_moves(mine, 8)
    f += _last_moves(theirs, 8)

    # Cooperation / defection rates.
    f += [
        _rate(mine),
        _rate(theirs),
        _rate(mine[-10:]),
        _rate(theirs[-3:]),
        _rate(theirs[-5:]),
        _rate(theirs[-10:]),
        _rate(theirs[-20:]),
    ]

    # Conditional response statistics.
    n_c = n_d = d_c = d_d = 0
    for i in range(1, t):
        if mine[i - 1] == C:
            n_c += 1
            d_c += theirs[i] == D
        else:
            n_d += 1
            d_d += theirs[i] == D
    p_c_raw = d_c / n_c if n_c else 0.0
    p_d_raw = d_d / n_d if n_d else 0.0
    p_c = (d_c + 1.0) / (n_c + 2.0)
    p_d = (d_d + 1.0) / (n_d + 2.0)
    f += [
        p_c_raw,
        p_d_raw,
        p_c,
        p_d,
        p_d - p_c,
        p_d_raw - p_c_raw,
        1.0 if (n_c >= 5 and n_d >= 3) else 0.0,
        min(n_c, 20) / 20.0,
        min(n_d, 20) / 20.0,
        math.log1p(n_c) / math.log(101.0),
        math.log1p(n_d) / math.log(101.0),
    ]

    # Global evidence.
    f += [
        1.0 if D in theirs else 0.0,
        1.0 if D in mine else 0.0,
        1.0 if C in theirs else 0.0,
        sum(theirs) / 100.0,
        sum(mine) / 100.0,
        t / 99.0,
    ]

    # Runs and switching.
    f += [
        min(_run_length(theirs), 10) / 10.0,
        min(_run_length(mine), 10) / 10.0,
        _alternation(theirs, 10),
        _alternation(theirs, 20),
        _alternation(mine, 10),
        _exact_alternation(theirs, 10),
        sum(theirs[i] != theirs[i - 1]
            for i in range(1, t)) / max(1, t - 1),
        sum(mine[i] != mine[i - 1]
            for i in range(1, t)) / max(1, t - 1),
        1.0 if len(theirs) >= 5 and all(x == C for x in theirs[-5:]) else 0.0,
        1.0 if len(theirs) >= 5 and all(x == D for x in theirs[-5:]) else 0.0,
    ]

    # Last joint outcome and joint-action frequencies.
    if t:
        a, b = mine[-1], theirs[-1]
        f += [
            float(a == C and b == C),
            float(a == C and b == D),
            float(a == D and b == C),
            float(a == D and b == D),
        ]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    cc = cd = dc = dd = 0
    for a, b in zip(mine, theirs):
        if a == C:
            if b == C:
                cc += 1
            else:
                cd += 1
        elif b == C:
            dc += 1
        else:
            dd += 1
    denom = max(1, t)
    f += [cc / denom, cd / denom, dc / denom, dd / denom]

    cc10 = cd10 = dc10 = dd10 = 0
    for a, b in zip(mine[-10:], theirs[-10:]):
        if a == C:
            if b == C:
                cc10 += 1
            else:
                cd10 += 1
        elif b == C:
            dc10 += 1
        else:
            dd10 += 1
    d10 = max(1, min(10, t))
    f += [cc10 / d10, cd10 / d10, dc10 / d10, dd10 / d10]

    # Recency / first-defection timing.
    try:
        last_opp_d = t - 1 - theirs[::-1].index(D)
    except ValueError:
        last_opp_d = -1
    try:
        last_my_d = t - 1 - mine[::-1].index(D)
    except ValueError:
        last_my_d = -1
    try:
        first_opp_d = theirs.index(D)
    except ValueError:
        first_opp_d = -1
    try:
        first_my_d = mine.index(D)
    except ValueError:
        first_my_d = -1
    f += [
        (t - last_opp_d) / 100.0 if last_opp_d >= 0 else 1.0,
        (t - last_my_d) / 100.0 if last_my_d >= 0 else 1.0,
        (first_opp_d + 1) / 100.0 if first_opp_d >= 0 else 1.0,
        (first_my_d + 1) / 100.0 if first_my_d >= 0 else 1.0,
        (t - first_opp_d) / 100.0 if first_opp_d >= 0 else 0.0,
        ((first_my_d - first_opp_d) / 100.0
         if first_my_d >= 0 and first_opp_d >= 0 and first_my_d >= first_opp_d
         else 0.0),
    ]

    # Score context.
    score = sum(_payoff(a, b) for a, b in zip(mine, theirs))
    f += [
        score / 1000.0,
        (_payoff(mine[-1], theirs[-1]) / 10.0) if t else 0.0,
        (1.0 if mine[-1] == C else -1.0) if t else 0.0,
        (1.0 if theirs[-1] == D else -1.0) if t else 0.0,
        1.0 if (t >= 3 and all((mine[i] == D and theirs[i] == D)
                               for i in range(t - 3, t))) else 0.0,
    ]

    # Response in the most recent round(s).
    if t:
        f += [
            float(mine[-1] == D and theirs[-1] == C),
            float(mine[-1] == C and theirs[-1] == D),
            float(mine[-1] == D and theirs[-1] == D),
        ]
    else:
        f += [0.0, 0.0, 0.0]

    # Recent conditional response (last 10 opportunities).
    nc10 = nd10 = dc10r = dd10r = 0
    for i in range(max(1, t - 10), t):
        if mine[i - 1] == C:
            nc10 += 1
            dc10r += theirs[i] == D
        else:
            nd10 += 1
            dd10r += theirs[i] == D
    f += [
        (dc10r + 1.0) / (nc10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0)
        - (dc10r + 1.0) / (nc10 + 2.0),
    ]

    return f


class _MLP:
    """Tanh MLP with feature standardization folded into the first layer."""

    def __init__(self, spec: dict):
        mean = [float(x) for x in spec["feature_mean"]]
        std = [float(x) if abs(float(x)) > 1e-8 else 1.0
               for x in spec["feature_std"]]
        self.n_features = int(spec["n_features"])
        if len(mean) != self.n_features or len(std) != self.n_features:
            raise ValueError("normalization size mismatch")

        layers = []
        for i, layer in enumerate(spec["layers"]):
            w = [[float(x) for x in row] for row in layer["w"]]
            b = [float(x) for x in layer["b"]]
            if i == 0:
                w = [[v / std[j] for j, v in enumerate(row)] for row in w]
                b = [v - sum(wj * m for wj, m in zip(row, mean))
                     for v, row in zip(b, w)]
            layers.append((w, b))
        if layers and len(layers[0][0][0]) != self.n_features:
            raise ValueError("weight file and feature extractor disagree")
        self.layers = layers

    @staticmethod
    def _affine(x: Sequence[float], layer) -> list[float]:
        w, b = layer
        return [sum(map(mul, row, x)) + bias for row, bias in zip(w, b)]

    def logit(self, features: Sequence[float]) -> float:
        if len(features) != self.n_features:
            raise ValueError("feature count mismatch")
        x = list(features)
        for layer in self.layers[:-1]:
            x = [math.tanh(v) for v in self._affine(x, layer)]
        return self._affine(x, self.layers[-1])[0]


class _NeuralPolicy:
    """One embedded ipd-neural-policy-v1 MLP policy."""

    def __init__(self, spec: dict, seed: int):
        self.rng = random.Random(seed)
        self.safe = bool(spec.get("safe", True))
        self.temperature = float(spec.get("temperature", 1.0))
        if spec.get("activation", "tanh") != "tanh":
            raise ValueError("unsupported activation")
        self.net = _MLP(spec)

    def predict_proba(self, mine: Sequence[int],
                      theirs: Sequence[int]) -> float:
        logit = self.net.logit(extract_features(mine, theirs))
        z = max(-40.0, min(40.0, logit / max(1e-6, self.temperature)))
        return 1.0 / (1.0 + math.exp(-z))

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            return D
        if self.safe and (t == 0 or D not in theirs):
            return C
        p_defect = self.predict_proba(mine, theirs)
        return D if self.rng.random() < p_defect else C

    def reset(self) -> None:
        pass

    def observe(self, mine, theirs) -> None:
        pass

class RL(Player):
    """RL-PPO as a standalone Axelrod strategy.  The actor-critic checkpoint is exported to the embedded deployable MLP form; only sample mode is supported."""

    name = "RL-PPO"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, seed: int = 20261024, mode: str = "sample",
                 name: str | None = None):
        super().__init__()
        _require_sample(mode, "RL-PPO")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = _NeuralPolicy(_SPEC, self.seed)

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261024, mode: str = "sample",
          name: str | None = None, **_: object) -> RL:
    return RL(seed=seed, mode=mode, name=name)


__all__ = ["RL", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((RL(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{RL.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
