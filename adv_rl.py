#!/usr/bin/env python3
"""Adv-RL as a standalone Axelrod strategy.

Faithful single-file port of the pri_dile ``adv-rl`` candidate.  The base
BC/PPO MLP weights and the aggressive constrained exploitation configuration
are embedded.  Only sample mode is supported.

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
    "c-mCG@6K(_jof!H`uu9J`hT}?#Rv?+86yKYV`DsaVEE=!zaqI`DY0egoaa1y?Oxq%vRGC8KY#fz|Lx!Y-@pC)U;e{i{^h^_-"
    "+%qrfBv8U_P77~*Z=<C{_DT|=l}iJ|JDBT$6x--KmYCD|L4E`^MCqZ|NL+N<6r*Uzy9M7=3oAg_|bp;PxZh5@*nQ{<1c^v&;R}}|"
    "Lbr69{=j!|Msu{kACLA{_TJ37yacw{Kx<Em;X=y`5*uK{nOX`H^08`{B}R>jrE8Aw6)hyf1}U#_4&2)hko+j+4j*7zpI~YH-"
    "511)#qO4Ywa7o`49C+_q?^&TeE+6@~>xKbN4>(eBZh)K4wq-"
    "($V!Fz3uV6_`~mxdrxm~_V=~TANF?F8S||(^=tLF{>d}i_w8@(@948nJW<!9-"
    "1DukeY4Hh`S3p09`mmC`u@hggFpP99`WcC#?<dQou6!d_Bq}@`WO1bu^QjJ@4mkDTld=fOaI|p>s#-"
    "vcK7Y``eVNDYI#0A>Cta){^DCJwzP8Z=>P5Sc`@42k1qY*Gv8x{_>Ed!ea&~j+xOOb_$R-"
    "8@7=BKHjn;YEc#xfzrE+vYUl;|vzNY}zNX&IIQ)ly<;M>;t=?Gr?EA$By{C7&^m?_@jYsmjO+M+|z1AFK$2VVh&aS^{`a6Bo_`|2"
    "yIQ5WA-&bGQ-)L(I^ecuQTVKA<_#nNmsb8gM)T`1v;U}B6+8W;&>wWt1ldGxcX)M~-"
    "7CYU`x!Tk=)I;l=>jV1c1vNdj_JQ8Ip7+E94ef}r&+cr5cl>kOJFhowB>wc>9V`4D?Fv4;X^XblM%q7Yw|B3;sb1fTXFq!1+UBbb"
    "?cM1|KSOW-J(g5!{B{4FUV~qp_Qo073tmhu{NhEN*6-HQh|ke6(?QX{JWK2M;&b#R_HC`5cJwu0KIqbhZBxJFaYDxR-"
    "h5g_{SG~A|KhD_^S*EDAL{$_Qr_?yX*2V=b1gl7{G1+leP=dp#c@}@UVRf@PCr{~r4t{!{b|QcefHENX{G#y^sC#`)>zty`tx{nt"
    "@+?&>QS|er$4D{x!T^H#b`Q8@jE}gqmT1Av~u&#<4vO%u7}YdXw4NIzC3M;8CvG<c%<j`_C9-"
    "RD=U2HRdBN7{LilasQ{vLou8!bq2>DS);`o(=*#P&ldDB*=eDO<kU{Ebt7*5NrB&5a#LLur>-"
    "lv4IiD>*=2OUe#J#3X5fMeZ@SL%<LL9_c4Ml{bGcmqT8#f<E%QhNuPLYm{?7%bk(_0u%-*RhRS{w`B^WfX-"
    "(`Ib;t|L7ZKom8Jye9&wVo6)tirU7vWr*7PpO+8Qf4iF{Yiny|247FFsCglDPIS7vK1Xp|k;UI+Yd`8IkJsqSouQNP5t3Oh!bgNM"
    "eQ5>jOQibrk9;T;4HN~>)@INz{~|K4KG*?sK0?=yb!_Z4y-"
    "6Mat((T5{#a!_4_it5ICcv!`D%T%ue78ymRC!mSlT&hiWA?C<<&kR7U}ejh=W>4MU}_RwF#2<|N7%UvTE8N+Cch^I(x(o$5y_(w%"
    "wRopEzUMBKmzCsyR5}{Mfl~D+Fm{EN$F)8oe1^R63!Gjk;DGtGNcWZ~8vGw%AbG-"
    "1F&_D2iSk$rG_!OTEV<$|~M#3F1ABH+BEEk1I^OGv2FSvI3_LpLU;4QS1P=2;qgOKKzhR=YWmdxCojbTz9i~qIT(MME3a3R~tfSj"
    "ypt)oAHy-a*3-trN7{yfAjPgPn%Z>L9Fh6`e)jEPa9gV<i=v?)E(U^hsdB`9+%-"
    ">dSCi3Y_2&YhajqG`)O;kIo9)IF2w=P+twz2J-"
    "&|4k!}vfO0M`eGWu>sgO8hs$MlBQKTx<*jOYK~E3YE$`**b6^su+(l5yqgEM9%n<p*s3B6=;ovD@+#`(fVI`PNt0F^pgNw1E8(YZ"
    "^(2{!G6~|4Ol2F};%&#C{~{(E}~L<K3M3XpcRP(VN^*-"
    ">vhdxAM5twU<wZ*rk`nzvTtao>8h1QI2w!NBAJZeryy@*5Zst+N0mE2bw&P7O2g<hO`LU4$4M!#f|vrrj^lMp~uu)?|a9|<|T3UD"
    ">$uOZ{0|P<4?HOGv-~}wH=f;T|$HIf|rlfh*H|)IybBNASpU)X;^b@|G2CazxUwq>Q>ixjl-"
    "^ZKtxeM(vDIN84upc7rAK_BwM5y+RJ)h%8j&vb*}V{;=0he(e<~Q-ixAl1Q0T)af#NXDe-"
    "Q!;k502tQALYk3G<dGmFcH>@Ln1$;0Ft(*r1sc=PFOvBV0=+8#3!NF{E=k)RQG-"
    "&c3{)qSSpt+=Z=mtRw%mrO!?TybBwZak+xK~KK)1d4}w8qaYCJazdg?R=3~^7q^#I!bz|c#qmJ%4(JFc76HCvh~093t2~k?5HH}B"
    "c3SAU4Am6wyyEPc2iz;<2D|LRZvjV>zTK5qoG9JVN)Ssy~tFxB^Btohsk+b9K}X9$u`s8PzcQ!qW!M@rjON^=h}8MuD!zsetdnch"
    "+hN&_z*_4j=I<5pDUAjhvZD3qW$Tcvbh4F#?YQq2;n^{V1De?uj}JS+#-"
    "aTtN+ly<tX<PUt242+4*|$D?;KRyz&}!OjP&8AtUE|EUQAk64sV4agi&L%_#UO{%1aXb(nSEII>Vj#4gY}XhZ6A8sEGUVqFb-"
    "{fc7BK3c{<WmVjRL%%5U$B#I;_3@L4w<B#B#4s(8PK>VGK6J71H7B={-"
    "bL5Te|#ZjLmWMQwC+6ZSO+0pH2TL^p;D=Qj{~myYWtXyZt7C>-k-"
    "zwY$Lv8nMO?1BkSyFqv`_AKRjFw+eccvJc_(aL2yVJRLw;euNzhCs@2i`r<Wf2>2h#?3X|GBCnAu3;v)zuR`vDr&lJBEXBGNXfW6"
    "#33eS4^+M<seE6$9z4;imkKtU}tcxBZ}OI6D0m@Ai$81x9jN=bCW$fF`#(Jt0W_YJMoBsYN0Vbks$WXxFsH?komWO}z`$8kdsE&|"
    ">0l$<T1q_z{=b#eXcr5-"
    "<zq8~^7u(9;rvljcfS|)Xg)=g`$GKtX})MdR$xU^&Wcx@v6<HNn7fY)2becdGQb8@h$uQN!KiYdA7q|183$YXU1xf%vF#5j3(D~H"
    "t~RW(FTNBz&^Z;~~=8f{gjL=KgknpBtf<cq1QVyBS<D|7Qj_`8YpN;ye7<GQfk@M;cHV8uP%|2*x?Bc^*m8Qmsn?LNqZKhkN`Ppv"
    "5VqBQ0Dyqrdzlvp-"
    "hPO1pN4>|=yY(Eu2i!;hgR}SiEp~}HYn2I4?Sn*Y~{2V2=y=%+7lBqT>#Ug35Gw_egfwE6w|04i8Rw?s7Y|25(9Z8~IHGj-i_IlR"
    "Y+&ElLTRF3K((_9kT+e!4dgj&g&#0%h3WV)<65CLvru$g+B+>F~RaWY1(1AZ}^vEz9sg>^cPmxs5;P0#&M)!aoPZ1=8?j~PUhV`+"
    "1R-9H!2(9s<9MKv&Db`7EWYyY;di;|&2>AmbfZmBhQ=AVy(;{`!4$vLJy%_IGheR(70EzGH-F&sA`gXdNwW<1-=Uu0b(n{B-"
    "4r&BV#S?bBPLvjTd|A;cfl?e)YOSrI&tl^>{#yS`Cq(}`GXcL4LcZRq0$r@59#LyeEG9ROlH~$Ya0fkpsj~|ugN>|KRidKduKTBR"
    "5Xjcq^N}2E%5S9=ji~FGq%=!QOH|SS<-sF9DuR+IZRdJl#BhC#mfeX1<#?n@RCEgVN{{r?dSOngUG!RfFce@~q}0l~btaXQX-yQW"
    "{D-__u4YAO?fhQAkJju1!r-=f{^x{N-"
    "K>i(KeDV+@~va2Qji?+Y;O$m7K4J?MFi~?$C|WBXO1*q53yrcT`)r3xx<J@Ky|u)bV*V<u-"
    "&qj(9YR<i)2;f#HZ+*Z@#XThv<Yxp|*XKcOF&OXK%SzfcF#$S;^h2_}9m4fjAYfCq!u^gpI57<E`=;tWh?`<)woX53Z_M_ngA0eg"
    "&y|q;vWR-HCb)M?d>TS@w9%&icPh`*aK@0iSrFFP+~{$3+jQg(SU>a(Qq86IXTMb?w|JPPAKrRuplS3gqR*t=xGyy_&#K0cDhOE|"
    "As7K6jwt?ozbW{?vbXFYB<8)9RZqV%kYCu9J^8gDoS27RRLdC4;m^A<Dt)WW$?UHuf0BIjEUpx1wH)<<i1%_PCPBwEe*V8sw-"
    "t<z#$uV07E*_$XYE`)c>)ucYu(&{Lwt>iW#+wtp1AS|2L7s2K=uJko6WgnCC{gWRl}LN<y+MKfg(I!1$43Ruv2-"
    "2`d9E{PijHaVk;1?uOX5Sk+F$BR?yvNIJXDv_=|?eT~u3cd>*U>Qkj2~z6nq7c@7Fetu8=KI^gux=va&ZtU|LQ{8-"
    "Hj<7|ZbZe0M)9o$%bHe6jJBeJKKIP8#M1xPZaf50DpRiwz_O}oE+kbuas0G#6y6kL^pq6^wc1)c$`_Ew-$RnQf@J6ms#MEMOR-"
    "H$qKyqEHS+y09;Q)-"
    "@PQc!2WR!M%nyWW7JLYhrDvEfi@d`6X8Hs!v(EqcBTyRcy$uGbMaaqmz)>$$SLmPujL4(lq1QqD&~Ma!aCqFLor`}-"
    "fXp=qx7SB>BHj39fyzL259{3Nd?lKx(58r~1E?sQc%b)=R#$(qa@DjV4NMjsT5EEg@bSn1WyBKz#W=2)Bt=1r>Xapj?WaAaO{uM{"
    "pdN|uV6QvhV_WJ?N7G^OI2349nR#gr1ued}BKhDr`Q^OngUtn}WTz#%sgg;ll#Zfym(#*d@<`63NU;FOe6aPp08w?v=!F>>qk3+K"
    "|ErCzS~RQgk38W~yFJ1}1UMy_6goPpXJkpOcLkQDB_gSI(MI`4uVm8sQQV#pXI#9Hrh`c&<gZBHm6JUhEi?Ke+VqM|x-"
    "@huI^bGXSUU1-KE0Zec$aP=Jx8N-A3f3s9is=-rCVL^`$nkd{Ao9m-"
    "IF%Y+!V^Sl!S8$t741%`miH)NAVX*27DNd_ET+Wn<y<D@%};ZAHW}Y<V_e~ncgA09weWcW+==kAJuo)rc`#As3RyMhZNYW=l-"
    "ESX*545WC{hDE<k7|J(nj@%<}>+S%2fIio>A8MQ*%+m9j3-"
    "$uK&k$JHa`QyxHk6|{;DTz?4@lc2Ce^;l#zaS|21NzjhFKRZvD;v<ExTW&zS{v>6Qq}yatqri1C9;HYh1zFGJP|K}VQ!=R};%27G"
    "**wY|^v@sZWR*XUic$YiPZ{}??=Lzr;PT+1@vjsAncUmCd#QSL@(%aP8^1HJJpv4-"
    "{Y$oddhY55^h3g~B0ZK9tY%WdYCW_FdsUl=!bI_yP2az2q3D+H^a6-"
    "4?Y;yX$yQkHM*6)gc{_x<>CRn`BxV7KUSJqfzs$CEcPIdoR33ss9yMJq+}8a9Sw+gK&81sa>q%dWW>4aK;M)2Ipu}xv$8dnwXyq1"
    "&Xy#wq{`5mXs<#;cNdqj|)ODB4F{1b4?9pN4E5tqe@jZ_Imwp+y9x+Eb!zQGIWzdC*jZMx=>cHpD==SM`-!uUUYx>0*zLlDS3+m-"
    "4QOH!}lGQ351uT8pU~&Qp0=S5G*<@9j&MT)Ktss(jCY>`qGbfN*Z_LaTcrUbq7HyPT3tr<<(N7g!IJ-"
    "85JHzx@>kNU<X_sHn#tua4Z$tnmFPKzHr)pWfKV?kx+w`zs^dRYW^8`vGbRkDr*DihBJ%<EciB}Y^`mbd5K+?L_a+P$dRDV30i>k"
    "BYgY<14(L?pfIayyR>(o}|Ytb+48Ax@7gPf|)1!4(=Pq7)~YS5GVqSR;Q6gd`=tOtn5PSb*EYtg#Sa5D6|-"
    "TvCZWtl3_&!odki8sDPRB4G!)<+98ql>L?wt-Icb*K!+%BN92GTo8_OU=m4(<vdKqMp&n0^Ow<O3Az;Z(5sUuY#aa?vL!Cbc($(-"
    "Go~$w^esnnR=?dmM>fHnALZCa~Bw^dvJq)1G<a|!IX}QcTvP??eYhwe(V^0)TxVZz@*pB4MI(l&<c-"
    "u`O2z}0Et(E#ySXK$ogrH`OC)RVa>s+6rd_^CNBE@3?fv|-"
    "qEB0Rm@TeF|wzwFEh2NHpymhxxU==x@`d5zid{Mj~Jwdm2jhN^|beN%{cAS)5haG5GuOYRLT*8Hf3L@*e4-"
    "dBd)mSKf3e};1Wbi&0&~SZKS0NPyXV=xhAKHzD$oU0smexv6C0|S1&^8KVY7Az?XP6xu2eZ7Lf9`)iNuk(aL3PFd|#xP%;MG5$(e"
    "7wSt=51)+Jpdi=(|oYw09(+aOqUB*SZK)DDfCWDKDfHt9zCm_m9;Lp2|mNuQZQ){G(<R4Ll4$1(xrh_5%1bBMHd>|F&uz7KDJ8ZY"
    "jCa6LJQ-`1r4yVf(9!xghm0fY`G}`6$s5)M-"
    "S7?hv^N1S((!h6|LuuE<+0|93KUb8{j<^{!NU<J;f?hkx+K&6AyH^ZErQq_iRHlEZELqo}0&fOk&=?)y%k{0jQWy%^@dpAs=@TYC"
    "aft?l<U|`_#Y5@gz(NPe_<4ztl4Vj>Dq&$=Goa{QcJh^v`{l(Rp`W2;2T}sb3)o`to*@FTAb|lp5V>|8e55ZlZOcOgVO|=HL~3Pb"
    "^g6lk=~%pM`GAj+ldvA79g+RRfr9Y^7pOTY*Uskn0Xk{NoeZ^#hl`M<jp-(2cV}h+w-B9Hat7cPDap__P?i_PENl7zF7;+3-"
    "=*}Kq|smwJ9^MsBa*#m0tLutVcv@}jjyb&5w+j`d?nM>dWN=uA)&><DS?g==n}14t=6K8&)57TA#Xy^g*bDJ8_*JZv`RL>%yg%Q;"
    "|FH5+4x|Gacmarx<SC`0OhgMe%lbSfYjcN0H(jy&Y09i${b=h9xw~qd$dX4OxTcn7*u2fnnkKg*?5<JKe*$|j;0osjuFk~1~wNM*"
    "5S}AVR~FS!AbxKKLkc4>(ODSvAsxuH-"
    "&Zbi8{@C<yKTw<118~JCVvuN!Vz4eS+ms^3w2UHIiuqY%{??E+HsG@1=hFsDwEs8Ec5#AP000P9AGIdW6ys>QL64fbV*vgtXmPRU"
    "G72%1S7PBTekw*m^IJ`}D1%Ob3GoE29;&W|r;>5^z<34R2E4bZJ3*A=e_i`Qnz|x&#`#>Vl<2Ub82Jnd|<D0>RGLN)Jnqy2x>(UR"
    "7`<u?GQD<oBQHrA$ggb}ztaH1HOkQDtbvsel|{hJdpu>r@)Mv(Gc2FI*>i3g?0GdU0rR5ih^Of_bkxTV!wwCcf1g{a+n@?rwH)_Q"
    "<(8^=%fnSdbi+p^gLixhsk-wmO+?+-"
    "e!E8~1P{H=m;*be$n3axfDf1w{}v6iKLWILS9#nOrrH@OZ5L%U9^K(B9OFkRy&HNf2ffQ)u>#EGUTJd`1Uf8{f$4622do-"
    "v?VOAkb#Af^J#!4o~4;o!W^To?GF^iMY)m1RTv1fk9US(CslLtW`BbXP>lCM|^)tP@D9NbhLnET175iMu#H-"
    "x|#3<_ohDm<7aO~Tl%jbP~*|^W-7{me>86=-"
    "@KRsmv#mDKLCDG0~FvsFseZ+zP+h9Rl0jYgV@;B+=qCv0au6_u}>jQKE1{VSgfxU8{bt6twUPC*?Xja*dV9S0uD3*I0yV!#~ZLXa"
    "~Q~w7hH_;W7c+|Qx9cnLRAd>PzQrbeiAK?lI-kf&l@zn32QGpLfj~T@8G`y>A-"
    "b`_C?k6T0wA7GueRvF%WB_B>q4R^!oT1IyZ5UIietO9sDA{q@ip&DFiWId^BjujW66<*fBw=7_fhVtX0dv^2u#Q{n;-"
    "?C6`%ds0ydRL5es!_lXt3ggcX9WcV!F=E{J$spvtS`Qt3XdDESym(-*8?Z*UQhv_w(o%@_o$-"
    "z3=ZtMG3l1wh{ngMwrL$?W>`hXAGQc;2**Vo|$=;8GK>N*%K^J;KoPsE%DD`xwGQ8)_5+#*sb7JT3i*#~ydOQo&*Bh{_`0)z$v1o"
    "e*X>ZUl^?S|jnTPBZ?!4_NilC<R8cqWVLUv_ffyhBOh!DlxWd%*r?F2OFKXzO#;Wy|5y<u!R#QIx=V)MIvTftl5F;~II;fgedwrs"
    "rTI_p${HS0!&i8AS*D|7h0UU5p(%<rBJ_j+eH0GNN?(aQ_Z+??jfaacD(2LR3Vp1H66A7}F(^F9Y(xfcyk0TE8|Sc*9D-"
    "9{R?~Pap*%yPXk5{g3s^Xoc!Xn@yby3`Gk@HGIJx#4>k~^*^DwtZI$I?N$x|xn+}RByp3Ph7^J>&-"
    "XimyoW_GN4lAbER6&+Js>qX@bCuou@j__t1~On!9P0`xG#hKb}Ug=PNYxjJ}{}irn$1(vB8OK&Am7PlJo5%rX3QBNziPR+{yB9u#"
    "x9-KGjAfSimGOlz}t9g!4>vgq&gqNjCa{@g6q~&&KbF4`M&^#)zIPnnK!t^t-"
    "e}7a{5WD1L+M*;POQ&&zufK<#~bBDGE0F}U^MQUy$`^K}eOKQsqDXw(fPjW73TNd`<nk;@M&(8F=b%7qXF@yf3|aH<1?hZ>!_;6!"
    "BpoGR)kwKzHfUzRv}T#k$K#pkwm;I~j8X`7MNoGSbDhM=n{71XU!S=NIPt_x0C;K=UWH9uJ2d$7Ngm2zmppl$%N^8!&{9P$sE@KH"
    "c0>(nhbhik*6E9!tuid9f!LSbR>gGvtO>7+3br$n#B6gAphNkrxt=*5!kxqhbiOHxQ<=F#klj7HapN>x@?;lF3&3Xn(#Q@dPIag<"
    "o#Qah3r5}Q*Y0xm+JIrKLoEDX0nc%jh3^2Bn3bG^9B?@Vl<<xtXO>N{-"
    "0=$0sl`A9W^sp3{w$k?3K4qP^Z*`lw?w<C#a(H5s!j06MFhhG|K<Z;1F(F?6Mu24A3`WT}iaiqKXZq!$-"
    "#f$K&07yqmF^8I=#nLotHFLQiMcc>(g+|qJr`rg<<CbbP5`9z;GnSe%P`O%Usy8+#**u375Omevy3VV>d_EApRFHhg2DCYW?y`p!"
    "dG#}om}u-"
    ";89G&?b*V%zt_4ar8t=mtzLonr$bs~!4)<6O2C+#l^f=rRPDKj^u}(0>#t*y&DphEq;RY`@VNDQLut_L`ClelB5ayAS+hzWy-"
    "Tnxb+H){(GZCI#Xdm~KzWi~25eT?JJ_i)j3oFj#TSold+{l0cA_ex#U1I`S(y^7S4MT-Gg+aZhYkYu4jYdTyTgTaQAq9gKSRUjlN"
    "Fy}s6(N!&t1qmVvwQ%R^XK7!veL+i23*?_41aw!Fi>-"
    "mZ?Cd|FQ_vKxoPBW=vaWd;DoMWrW$~jVF$<7M@8C8IDmZ%=!ga&Ph6)3s_X(JKIpiRy${e4Qka%d72imw;rbx0Td#-"
    "B{UFIuAtpCUpGw(5YC%}b#-"
    "nbB7HMv>)+mLvkGSsHYzZ*Zsx?T>8wH98F!H$L>icot>bHTSl8ahc0(mWn0LYEF<u;dAQiaGHxzXW81k$Dd4!uE-Pj_#=0<X-"
    "OqoGgBAOJnRdhSE2knv*2n2O2ir6B@(Q2ia?8Mf@(2_?W6(51w2nS9k&-B6evl!vDW)fH5{2SbjY-"
    "3k;R6xNhV9zHn3K#?$L79hpVG^-zHKoBofb?&_LKYE*lZYoUJe7jl=P;Y=Hs1kP4ZmuqVn%@)JQ(hRl@wMeUxo0$fz-"
    "u8;TOHW5v@n;D8VLPCkCY5ZliuYeW$(#|g?6rA&`F+ms^fYibE9@2bjqUK?F8Qi&BC&nB8=%KZ15?#%HrjqHlr6{wNjRXj|g<$g^"
    "~ceoZpX52Wi&RU;6A4X9m14gjH}#ve`hleMX;b1}hEK66{&(*q6C6B_HY$Sa8}PNjcRm=OD)0nJ?SdW`|IzB8B0{CTvYq2vb={kK"
    "Na!t4M`4ID0{SM2_QTAk>)rhzB=l8_m^s4S3+u-h-"
    "V%zo~RNdMUtoT3<T7b?j860^hS&X;j4?S8g>RHB$79pfo8p;eOD<^1faxK_7s#LcNlUmHJ3<a6|ZP=+t%buK59Ij4gX1<hmE=sUc"
    "9yQWH(7$+8s6xweXt+Y&hEmx)#!7#tVz%Yz{p&+T%5lLJF;nZHKh+g5dvujC1#z#+i`4Fxq1J}fU59xhN~{3M0&fk3FA1gkE#sZL"
    "^tuMQI;;S-"
    "ex&m{_qtb_^ag%bn^`ja!O?V>eQoN$jM8y|2!hWQjG63&3otja;OOItK<WyC)u<B;Hc>L^%3wjE^C?`E@6&$~FYRDiSOj!5><mVx"
    "8774V%YxG-"
    "Nf{omwwQuT5;K9j9i$P=uqA$QPA8hKy`x}gc(14Qg**W@{Y-t^cE%_c437$}CFOp+AeQIA3o{qW{iUXIWX9vL;z%I>M|do<dVJ-A"
    "a4e+FiN+LAyL)xDEcatdh3DSm_Pegl%A45<%+*SEMCkGY6NsCX_B{;q2NjQYR@N4!LrFRCrtb;M#ie>V&9YdsxlEU9?5HWa*+;gH"
    "F|0`DbYc1siLP=9f;ElL%dzM!rn($QurAc|S!v_09tU<c@F$o;)p(07=WuT&0t%PdmuwlsJn7fB5n^*<~)cUSTX;zD=BPWL&zi(T"
    "o2mrPkD)VGjPK%aTToFPb{S;Qc2y=F{OSLO`}1_G_l%!ia?Q87b5bE`IsbE;*CnvZ5;{cB}9N^XJ0=ru$w@=-"
    "gJ?UOmYk|=b`Yo;G;W+!{X4TC|$9Ws-x#|i;bpx>TV#`{wbHLC&Ic(;m_RF|&zN?GLRxqyu{iHlh^zw-"
    "}iq0b|lYZM8an47d1pi#1iW^QFEMXY5(>UfDmDe^|;MoIt(PNonTH0=;~(#aB%SIEt6eHJxn;%yD=akvQ&<2b+M+1?x5T=&Afpx="
    "GwbwavMR0ICwi>}Yd?Vpl2=#>_E{F&WZ5y#0}hVS$C-*m43p^FJUh%``FEx=Gb5uLTQz-3M{2%-LbEr2>2-"
    ">HhS5S3pE?6#6gMWj~>M(>7RK@x=ysVi+5laGqu(Tsq0LW6e5NnrfI>`;9Vv95k}pk*6~(`f{iVBa?c7qKqYlPL*_BB0Q=up;4H+"
    "Kh**=iNLmpP6Rv$tZO~7*RrL3U#A5Fw*m`A!4}bJ#@wsRRICx&V@%71p&kfGiFINPI3mp5Rjb1e@HpFF&_?;Uj3<>1wfskOqzpLL"
    "qL@u>c)vUf=*Z_krXYo#+0zOh<RN9z@s5cd_{`e$HmOy_C2A!SQJ$%>>#kWo#ZB@<=i}aX><nj^d7qI!;5!QY^C=CyTeRy(5#sx_"
    "KSqCQ&6{-weYg>0l=$S-"
    "KBKp&*(cK+s*7i3q+4t+1w>TDn^ekB0ZvkK)|=G&N?)C+G%~H>;!C6?;mPBUFn_JX=DL_v}*>4cTEO*KL34{2gZXW2kDnD5ba0$&"
    "2+go*-"
    "3O;(88{3p9I5BvDKG}((s;;i$qz5m~0{?f})k+@}}Jbd9pzdJBep3w`NV5>Z@dHwBF5UBRSf+D~BUCpqo~{1?S~MBV(Z&?f|DkVW"
    "SC4Z8w|=glf)Z1brmx%It{3Gk@m6>Q6^jaEKQ508}GN%T~%>28~6^3+Kamq-cF01Hz)>CC42x-"
    "vU@?D;IwNZxSuk0eD`)QrZRyp~9On^QuV;e7(5o&5y{ndvtO@;3*5EY8l4$mA2w}!PS<88*nmivJTxT_mNrnrkc=dPbe5F3(QayP"
    "1y<B5Td&q<&-t)??4d-"
    "EJ0@f74H$h0?Kh;;E@tqt9t?9p!CjYwGLF;W_a5>VwJZ9CKtNCmYFUQ^sqR9k7BLfjmB1^#7B<zW>J1;x1iB-qJA^`vvwbHw=##5"
    "w7-"
    "trHFOqQ^Ihskme|$Q2x$S?oBI`rs})FJ63Z4YC=^t<WsG`m&|KuXbV*M+Hc(D(cnzk&rSgyPKr|#D3KF_C>JSHwwU4r|Ih2M6nDk"
    "5fGnoRYF%qw}U*fJ0hl5Q@u*(<XGY_P}18UjM>zo<bWiX|>F9f`D>?)(<jNUSCE}9Mt%)kA}#>#=WsZ|3sI@Z?m934(tE#qga5`q"
    "_cvT!R4+=3;+yYr2-"
    "6JZQ%2D~_&b!UA=Ns2)N5{TS<a35d+!#g>F@RMjYBTO*yM#7Vr!=T1Q&i86<XuVJve(`5UrbyZ+e1(i^C{hq(*_Q%Ty|N$3=ys@S"
    "Ns?bsk1^qZr-@xiF!?ABv~mcJvXmJx$BA|q-"
    "gWGGntfyjTx*v7Mn4DA|FNj%B!EgIhBDUrtCM)bp&>Yd2?zj@xkIH`oOESUnO<344wG{Jc%jg_kY!yw50YGPT$^sA2E2wq*p}tzP"
    ">{iqLTG@7l+OG`8y=mu3vBZU-MB975@J9Lgf&L&EqVzj#tmPymRG9=hMvjWnT9ScP`_2$HrA?=47ML<?r@)h|7Gt{p^knNHTZSzU"
    "+rrAYPK6|b=)ve0?<Hm3Iv#c2vuMX)6G{1FRdydG3z8KFIp-"
    "<iVZgH0Nw0${o+wXuZkY){OYi)xM7FG2h4QdT7MllJ><@8Phm`91!NE9c(yAL<3fzFT;ya&{j<h*S;n7PKghl{X&+}PWja*wn=At"
    "_mXuQ<eWN~b4jMsN;cdZ$hm&?#@X%;S@V8#t6tTF2ID>#Yx^4-"
    "SS$o2zq`5>l2zhHr>)G*$9WPCn>>NJC$pjT0i1cFFTtCMp%uTutaG%wH(0lJM|1%}dMns1$Ov-GZ#Tdk(Z>OipAgP%s=Apd@M$DB"
    "ktO$*pEE+`3#HCGBeV)`TDVswtLg#4*!F&RE&>(3LimHtXtO(_-!9E%l!?ANx4)dHJq=Z`AUYkWnT)(B*<nY9>pTd)(sFYY-"
    "7@jk6GQ<5K;<|qp?2V|wO(V8PmoDm*iE*bL)WrHyv476l*9c{&0t}$?gJTdEris-NR`izWCwnFDPHHm682~gGdLsSg30$qT_`!^m"
    "xCB|EQTkDP77>P0%>DrS%><|PD#P>%fxRu0V%cz`AjE7-"
    "KwcmpL@9buhE(!=(#xibwlP}(KJ<C1tU33ykj~FuS2t+yFF!|dTwBVtCq60z&N9^qivex8u`2<y*#ZlWOf>rp7<G|>xyri<S8=n`"
    "3IQ0`-"
    "zCooTOtJ49hPyPQF96u(>4;Ym3_fOyU;cXm_|gE1yvaXonu*P99{agWZ_WxlDT9Noryqv$AZ*o#=4K)==h|t7Sj!|I3j9~raSaHB"
    "()I2N6P(89HVjvluHchv-rgtMA{lFy>1AAk1M|Hmj)RSKrZFxS1$~xnRXrO>>kMUg+)KfVv(}MtxY-"
    "v4V2@vOOLYy^x_Mnd#;^k@!dhptTQ!>BK{l`wWUYh4*(XQ>6v6;;WNQoIP_D9Bs1_4v`>Q`7sxclX)vux#5i3S3@d6OWc3i>p+BD"
    ">@OzPUVNhur@F|l&2*LP^z#z)|eA!lkTK{6{6|#K2I*vECH-P#<lEgU&DX0sA;B2s&7E&Zj0~3}+giJfh)-"
    "AD1NCT+1wZr$XJ|_jVd=#^gI1Kz^tl_BFSDGNwWo8$0@d866jG#lkAeYt#k5jYE5`$LxMJb~r3&aaLanTm$w#7$}v##4EnW|RM06"
    "<Q&hFX?u7sUwnQu-WcHf?%`J9`3Cs>jjIW>$-"
    ">7%M+)h~;rOBfkys6ka)|inR9?X@fvQ>Q6TXKxU8f5d;%GFy4`SyHd2o9QPT;ZionpP4hOlbGKJA9N0#P+&_9uD`EbIJKnX=9VS%"
    "?Od?xETt&{ow~OG*3hK}_P=q(iB8Zqo=SY9`RVxhE_WUgHaqgd<kOQ|HnA-SmBs&>)1$seTGYl$t_fJ^ylm5!1^p7wCyZ{0c$xC+"
    "IVd4<5^!4UR!!CMw<JroAa~aDNjZZvp<_Tf*=f0Qn5FU~^A3E93sW2rZ2t|F(=Bf!<Bmd9LM(H%S(faFbG<8%iGqm^bL~joL;R#%"
    "j7nS^G7YOV&s4Sz1DU;Nhn~dD5SK|af6)r(U7y1RIT_9(gh<#z61Y!><VwiN@APxox!(u-"
    "7o6vvk#F+_hBpbLlyit@%<ID(xHxi|Ul>4@iTtnad>J=ZPumeCpBLaU!=%y)iiaP`|*uBy1ml?kj^F2#cLCSWB9cd5;r`diRnK4d"
    "-<^#7PfRnV@a=b#}4z!!zTi9nQMj#X-"
    "8BIG8J&t8m*v_TH(ZEv;2J@`DwSjwq+i54QkzQw%QHYhUgF?!iIzR+r(Q3{c;sZ3=I1Sie<?q<FAxHz;%q-"
    "AGTo+d|Bcb_jH_Te#FjRm+79OTbZHjX1Bw~RO<Df@T<Wn$?tPS^}$H_a?RvAgL4{-"
    "SlBQt?Ou#`*0r?TE<g@IQBDrywyH~b*sl<aAzm$iSI*UvB4ltXy8H=S8*QLyX*0NFqH5WqJUnUGE2kBrO9ClJD9Iayr|?qJMytU6"
    "EC0QoSiV56<+>o}I_3lubOCE0UgI&+Y(Z65Vy#>(eAVL)?|`eQ2e=9jKKB3rZC`O5j+y`*!LH#9H9iUGDq(!n9Q<LI(nVMkftX;~"
    "TaBU1?|{eq|BQ`LT*q|bT#!3pic?TKvaBIk-u>I$_gM7jy$&|pnQ0~c`B>|891-"
    "Y((~axLd7EKqimFm*(nK&t^a;YiRRe1iwE*Kd!^*0dv-"
    "@WZsHx+Xsks>5!HpbYsSez3*ZET>_m<fLxc(Y7+Vyc7`_jlRVDux7oK;%wPnqz5RQzC8~PmUBe~*K7_1*6Qjb>{u`rLCJ>+0;UVq"
    "lYYa-4Ig{OcmdQM?1OEz@v`%X^AU%s2c0N;rWj8Eq1%-"
    "tPajp*G`FIonM}GIs!|A;HX)|pQE(U_?ZWenCCP>cumT;^WgI6pA&0(+wfSZ9fMx6Vg?kF^iKMI05p?Bosy_r(k%R&&ed~$RHpAV"
    "p#vv=0yWgSrOa%{p>6Y$cIs*r%OymV4X<i~IFEvpw(KiD2s1JK_r6Hn1Zy2bn+RsRftwr?&lkEkPL}W$6{n4ZW3Q0uR;0q$vlU-"
    "!Gn0S}bb8DvEMJtNL2gy1X1g%<{1hQj4-"
    "Cp*jw83l~%}qs4@N<BG<p`qzaYUID63!lEh*$oLKrDx96FfX4Opgi=>^f6i<rR1|qD<p#Wbg7C65;xV1d;E;PL(>N4OAp>I@<<yV"
    "=O0879_;5Qiu+Yf<E{MY4_toMax+FU;0v%<EC9k{n=dba23DeiBk1ZUmIaQ<;!7D#F1VsWz2!^6em@g!><#A8%+0-"
    "(I!vV6}yvy;Azj~#z3fKt;Zx>pYb!cHZj=seSQuQ93^&-l6<6K4-"
    "O~@ZmBEa1{xU${R|_B1hVPb0yycSoPUge!Yc>UvUBWHvHtn5PM{3qlA&D)N!cKI&?I-"
    "by(WoNtKV#S1C=q_2G|l<4x?0!;wOqi_9D0|ZRHeB;mFhsf!cU7kxjS06md^q(JUgdIsW(4c6}Ls8$A~)dzSJGVCev!?iT)9w^+~"
    "Ac?LeXhjYDg0l;pV#Q_;f$l1!8hda<qc^7?J5Ph`RK&!QZ>?t+zd7v{@=@9L3+F~%XO8yH?p5hSzF}!oYSnz%!wPTGENupl9DGo~"
    "CVN^<-m}UF;#QpU~o*h<tb}Qeo43Qi2{Ahx~`@z<6*k?NcCoKl8P5c~RHatvHp}DcmBF=cs)d-S(sa2}qNPh=MVeA-giCN%_%>WE"
    "q3kg4qXWK3Nc%1a7Wt)-VAZxf;$#d-"
    "A31gbOYmY;sBVlq&uP#ibrPvPJe{I1mNMIF!1Bn(qlvY{*`Z@3zY>aY$LU*Lq$1JOt7$^89RjrCE;!V9YMczKWhiL+&nBROdMCq^"
    ">ZwPIXW1^G-=@|*I?;T=<@ou8&n0Og>=UQLs#?yhm$m8I}woFNMU`VuK!-"
    "R)3X{`yXf5*u|if3@h9=9(|ERKDXuw!a7qAspibJ}hDL9Pi56bOMXJ?>?3T|v|wz}~>-"
    "(9$A=paMqIN#Cm6@6hNy>niolZ{7j)(>JXf%(C#o$kO?H<|p}PE0mCeAEl4I&sQ=E9-MA=5<t?-"
    "aMNi;ccaoEm3dpSUoh`{u+f8@`9*%rgVWe3cis!<e7XBLTS{i3b5rh7Yu{o=Uj<Sec!0-9eo|biyBbT4jT2%yAll>$9wD^gQ$-"
    "yj%HtMp^2)H}YPQg5_IRoK(_FJ<7z8t<D7?A*z_L))jzos-"
    "o4x3MBE3mE1X2e)tnCSu^$pBUD7|!ZQw^`@1@~k$1uc*?vL|!}t^e5BEuUywc1F8uW_O*qVjq(~?F3zE(Pv;!E59mTke<y71QO;i"
    "-zBYzi=+iSmUtfSYHP-lJiGq|BVmywQ#XQylpE1DtA(r=bD5nHN}GW&NOATh2ED06oA(}^p9Iou)6YNxu7_UwGQ@-"
    "!2GBN$Y8a5t&i)&#kKD(dCh2FEG<I?2ZcY}6IXZ7SiW}CZarm$f%_B}L2WS}I+5#%!Kv>^)&YzA+JQ<oZ5J~&9WTMSZ2SKOT6s(%"
    "<)OB+Mx#O2ab-V;^PP~$5<7l%iD^h`xIR$SJFbM$UyJ<{3d9C15cmk#<zFDqz%^9JdE#M1xR!-"
    "^!WP3GtflW<T9KNxkoOv8wt8B?I5+w@mxm{cl0nK_r$4(Rw8TXfMv_ZnpwgbCQwhflA`gk{HR(gINNC)(T@S!fW!=o*_RkSEhPj<"
    "MwH!@8xu#p4^P{7b>ghyFI8_N%Esfzx#I*qK9zJM{~s)rGIK|6u$lKz<pYQg4~{i{b3favnkki|cRQ<*;Qi`$A05<;ailC)VNjk|"
    "6M9zf-A#f^$4^Z_mEb7C2Rv{xU_PT|m3kdagy=Gfbga_QY_u@Bpt8gi$oEx=NL3|)k>eWg4AN&#I-"
    "nq$Ghfja4kMenxfN)nmFIBeM-"
    "$ag7jYP9;$$zap#M}OARY6Gz4(yYO5IA98tRxK^<^;{R899Y0j#Z9v$4)_gIm2^|@NJ98+ljh8wyYQfEA6lUC0G&;upvuBwzpX4K"
    "po308e{a^f5tns<o{YiVlnktR8$bl%q<!Umf{1C6QTb}Sm69nmuH*w{bhF_~r%}M}{233rDP(vWXrpi%5^{V^a5h=Fu>f;n8QE6q"
    "2KN=>@iB^4bQOwW5*Vz1`VTpZY68b=`-"
    "y$gI>n)nJIchNNJ9KIFpA=yIEfmAeNOvm&5@NjMG@q7%sJ*rg%t*+pghkYWt*u8940bgCDhWC2}v|fKqSgD(m8j>@ngmuXE=kwhL"
    "Xrah5D6oIr%{Igb|#yR$hAJ1oiW{8IuSCxd9LhLR?w{sbyUBkx4rHdF5yQ*(KbjEHuEP(aBs07y;C$mFF|F1;K-"
    "t?TfMw$5PZ?@Dr4G^r{MH_@Lm*cUv~PJ<p@rcDi&dpf>h@1sROZODhwp0{f)RVv>*n(s^KfNT8m0GT_dM`AazzlZ`gpAVAazO$9L"
    "r!X@C5+YU|zp`2KrH1oKF!TE~mj`Qg+3rcZF&5lV3{<qj-"
    "9M@Yo?8Ntlp6C`njIck=b?_g>OmM#9EqB0x9fPvIEc4+VU@I{(kcS$a9#59mx>@5i!@Wa{&70NL-"
    "MkCb=r-zw`jAs3+TS@2Bls`LHpF45ez-Np!6J3Ls1}hXs&44XETVX~@gE(7V4rS$RUq`x8_R?VFA)ZiEW4{DR^%}O?9UKA7sg<r;"
    "&H<kxk8$=<Hh7g?F|$J1riV$;Sq1iG(!O?GKIQS!gx`S9mA5|>ZN>iPB)yL&7|&7k(lLrO^1w(h2}jfOR$ghOTE4cO;88)&-"
    "<`h{teX{<xw0Wb_%^Un>1Co4P}c`FB6tpX;?Th?&}YPa$maFG_&(zuhY5Mm7i==jnM<Omv>qxlU)QyWYbBsAqC+|2k=HNQ!;L-"
    ";}0G<O7|NS2Dh7V3+5tdY1)UpgH{y~P=<0(jzSWI^T8~0D;iX=p)dq63uzXTLy$Ym$=fQ!SS~DZU3ye8<A}s&BhI2<fzexXcX1yL"
    "zn$_hWc5({5Wx}C6f*}4g=OtqK}#R>WqaeAw4A<yR0ePfkYMM$C;EhV!+j=u)I_dBq4?~2gK2EC>n?;OeK^|vJ%9atz+UIc3JMYB"
    "3FaPygR##kZ(38&nE8Y&E-7xd@n~z&C|)&oKzWiqM-V>e5?Oa-"
    "!$MpavJ3~~^n^X^&RUwoHO8F*lwvHFxwo%2EA&7jqDj9{^*x{zn_`9Z2(}t(xKWL1`fv!3ft|ms-"
    "e@j*a0rC^2Qw`UleRBkPSzLBf-xP1_A(rd<QX=f45AZbq~<PW=fhhNp)RBtFC*6AVCKyKFf0%c9u5&WG6HTy3f4)iy{5E|XUMHd9"
    "ifXhtS3Tqj>Zj8ghJ?jX~n??Alh*zGEgt*0<L2{Zy=k{Hm^53nbN$FX#MSU-GGZ{h9_pH;U&dbh*Wagj*3<VQT*mHgKT1vvXxhUS"
    "y&SQ(Gpla(I;Zo^Xx3@KQf-!x_~&8P#~&_R()Y%`I)8-"
    "`Rb3ncIf)idV9$WX22rE<VUkM8`9nr<Q(LA$tOb<2`DfCpp!?3WS_w}b_z>#cmxzCjU&fwzIubFIN`BlgZWjz1{7|>d~2x^2TEa5"
    "lR_0N_E}ukYl9Gs4WmoNnX<V86ibAgxt~xjoBp@Mn-qI+tZ})PRyNVMV;<cS#x0H%nKt~l^h$WwH@_RPog)9DqfMDrhkplYb-"
    "`NP!Png3h~vQ)X($M5onE~T-2N&0D2AvHfNtvxL$d}{1Ck2!d3s_@U&Qi@yu5k}kPJ;{#iy_(S^$-"
    "U%xQawV|J}Ds&rnAwBk1>4%!YD=J8H!?Mj1+&VnNO&ASTrx4sEXOD05khgepCjw>d(@dCi2anzc;r{r8w^<qHxMHjd_eT)&J(?v5"
    "ja%9s3afgE&J!kRQ!%3BC1d2Mvmh5=Jfa?=NGPJ=%Y#w=rs;?!y&O^eYqyrd0F<ZOaNH87uuUSj4IVH;pd>}|DUC?cyWQzxc&c+7"
    "g_WiZqEF(3L1d6t@{Kh%ank9?<Ldj~=eJ}t#;^&{XvQXWjRxpz;GX}$x_-JD@pLi!;OKuXI#+XVZ@W^ALt~Pa&vPnqbOOphw^5H%"
    "<fwA{}wMH-"
    "g=+{RMNoq9>hs_qa4}LymRKUPPCe0M8HN)$GZm9bN*<ptWK+?)7%h5ehe&FL{;p6sO93`%=YtqsP)dYPvdJ+o)v<U;r^K6ztHXww"
    "1YgnwQcreJnlFZuClp~F=gXbQEc<7u^12VL)mxsPR+!?&E7~L8tl>W0%h(~i(0tV-hG8GQyio(jo-%d6MaWd-"
    "$%S<?U8=scqCfKKkNjI1-"
    "7f^tIda#^N;yBGyIsv{9sq86m7Wy%85Q}olo)(TAnA{=q=E5I4$RRJH;k?ozQJ?76K3<k1Xc^CLs?Swo?`fcJk5&e_P;~7wDx`O-"
    "5QEVhinHqWLO4SI_c+DHH~<``;^!S_y$=@<zzLjXq+c#I?gj)V?35D|BQ>E#Le?CmiZ|nbdR(CAs033K;0xPa(%Nb=^1ml+5u6Ny"
    "g=5PnLx=Zg>I)V}{51zZ_xt)HwZ?@Vd2&gQXd|r8&fn>v%TGd3>>i=niWJxZhG^i@LW;n6myJ_!JlUD-GRJn>Xx@WzBOVwYU~1~M"
    "-*<#dq70=ho{dSTVl#F_JctPA%mkj?lL>*=mX}Hcsx5X(+uG`%`V`UzsthbT(@Yz$Ijo^Z|INvp+*kItO|L+l3_+F*h$rixEMwE2"
    "!bm16T#u5|!*~)Yji#u}5D@@a0_0Z%_C{Ss7RigKGXhqKDGi+8GTJqJn0f`uN)kn5P>53&r$2GwO}F9UUH-"
    "{&Lgsm`&_P8IaS&V<2V>7p1LnCUzqdiyu(qsZT`(;^_qex(v1yfqK>UP*=FRHYK4zGFVfYf%P%|bCsa?m&D8;uYGoYz-"
    "D(FQ**FNkdWu08rY<NF0g~LtWo=loeX6)%PFYQNIa}1++@%Bc_ikRVhfD;(^=PTkaL2^v%wm5P61FmdfA}jODPIrfO2qu<Q8RBOM"
    "tb}1(D`xzp3u|M5op#|?sGMa{0r-B(9srq<5V8MfS-"
    "R44H~$W}Byl7h=V|*4k0y&nw<QHVFU)^>nMRu654Hnr5TF?M!F09N3R!~R%iKE<I#UM%fL+mX-ByC6VEy5t&ObzQSdYjQD$W<#Sx"
    "feVDFh7v+YX9Rd2Y5TjYCM}0sZzMUfghP$=Z=6pz?*YO17C823m7&Butr;`GhlyCvns2FPSsCZt~n_bD46%e>O}~{M(drF`^QR8j"
    "DS3mU)GFM`&G#|HZC7z*1V8>U21V8*+f2ow;rmfRl-"
    "*j`TC!fEvVJn?_;(e^(mo;_JxNcntq{$>U%J&^V#RRC6q8oIOqyGsZGl4R5|}SD7^Iz;%j)L}%yaU=n6uFY=+-"
    "_^qTB8!qu$)?c+E@`e>V=;(8EDP9Cn2_1l8HB4MjOuKZ8mXKTHFX(MwH<pnVr;IkVp?^Qaj=SD2%{H)Uk*UDf^GuV^T(m56BHhD0"
    "pWhtSo>kAf(kWAs#QdBP=yQD`Y|k}fIwyUjFiQ^$nu`k@zBbIZi_eDA;P$hVg_bGKH5LWqsAs#jS-UfQp8z(Dl(;xNcqUO!%p^Yt"
    "ZaN+al@I7T-x?w>UWg@*NB?XO!AU1H+C0#!Jw_*Gpzp-"
    "^88+rV)#t+b@>xhjT~+=(xSkzud)Ln~k`2`(bro{gQG;6?_5N<yA{iZZ@D0LBM16)i(h-"
    "0Zczdo_cMb)I9*%fTCZh~^oG3J&0m*v=#4@t|yKry;RV?jnYQBK04X8!r{I<%<+M0W|;j|0IKWlI@U+z_|=@)E%ys92DS>jfN1ja"
    "hD-(geLjC0nmnOM*aHKs)fP$SnOEx1{8pBCOVP^$q_cTHob3CoZUaP=}4e1THn3-"
    "p^vOjT*^k`=LWufR}UObi|U&*@@l%XlnLMcXGw-GuSz^!&x5y1}raVF)S!pcs4zc#G5(jCdc}JH=HL*)mg5QmqAmb0`-"
    "*)i2$4aen0CMwo1mU&>6EXM4$ic@=hpOk}t1d~Mxj^~N@H?~6nT?hE8n#%B<pa!M^-gbDi;>R#s2pyJQ7UT#gurLoZMA;oD;a2-"
    "pnZI>Lhnu!518V>S26HHEyXb!|3JcMVeiUeKi@|f^fw?617U5dCCP4o-r0&Luv-iI512puV!kG-yM-tRV7*-"
    "#n#v1R{A{E4w*_>`c8`OVM*J>&hF(2}wY8i&sUi1r(=ibfjqKqk$<G{iCOus@qv_X_&nX&B2nA4)v!qhIkxOu6_;_3_ff;N24#X3"
    "0?i=io9ZD&<%idpmK&mkorgD+pYV>##=()^s);RtG=@vfNp{!p~wIg^XM6EF3nPeKX-nChGzJah+oF_3wwafZ_3qa2+|nQO6`jK-"
    "tG**Yi&qx{EOBS*$sKgVX2*V>p9I`sdTZ*m5>9&Y^jNJQf+pM09M~g|a6lwY<fU(Jc3HUK?@J;K*d8L3#YV4S|&+DAUdlKVJ;<Z%"
    "sb|;d`FYhfByT5hVlp2)`7wrRZ>B-"
    "T%R2yoKpGx#3LrDC#6Kd!|CvR`BFxCvYC5_qBv@xlSOV9KJv*S&jB0yKlFO#^#A{OxpD{i%hPtY~VIp<0L(w^qH{{$CCncbH!wcN"
    "#aajrsudB<HDbZJQk4|XD<pzvmF!RgiS*`2pR?bgJ?wYn;3nAHE$9(5r=v*cJoNUGNT_JV7E;qe)$>*>6|~*6}8!|q@p-"
    "RW>%VDq!5hZ0!XS!X>{q3*vGJ)bO`}_G}WU6v*&P1^opJ^fq7RCXbHVDdth4`g`Ok>cC%z3QR)e{N;9d_GH{drdbp@^_o2Ug_7e)"
    "g-lLjX=FAH_!YR)~Sbt6^p=N9N<xFR);ctTw3nA1R_Jzt9hFk<<i03-"
    "z$RVqsFq8o_3X?_9?axM%U~fq=2f^WeV=?*Cqm(8f;y2n6Jr6kKL|kqsHvGk$WMq!5_JPCn6e$AWR%Qee+Db*TQ15UO=;px&oyFv"
    "Xx!5i3!po&*R9y&O7`aVA({hwyO29H*mROn|V6@2Eap8dJPqr?fvl&<h)ww2THu(G5s4CAW^h4wpwkq0bAvzrvF@k}H{gngoM@!i"
    "LTqbjx2}|a?Ja2dR%1VKIFwXfrOfi0v8{3>2D!7(Z7bcsU`%UC+1!W%-"
    "5bWH`@ZLB93<YN9o!)X<iQAQj0KUT&rsB}E^4J}WRm>`zsCzieM5L895XvA-"
    "!||!exlrhVnkRFCR0rYK^O|7CV;vQ?Bjj!75Ma7rh$*hu(TBVkcBKAzV$KhC2K%huBqGi|Ji-EXK*o*fsB`Txo)l?H;Ma^(annV6"
    "63q@2Nn6w=HJrp3i!(+X-"
    "hHdCvg0c?4dk>SP7q8WgaMK6#3n(BPf+X_OBrU`OasHB)y^;Rh_p13s1nCsqKK4!<U@v=>FcKJy@yGGM`0|JpMo|IO`8?DImm|=A"
    "(Bn|Se2JO*NxpZrzvO&|03mRH7~Y=Y3PI@*%RaPpX)n(b`*e>{v(liM$}{u0-5S915gNW)aBog5k^f-"
    "lYh{S40OdKcj5mr=sLVU3$Z8B??jpSI=H=dDt+*nRoqk>VKz?t0>DD6i*#U_<X+PRm%UN!iU$`Vx*1ISM5I9=t($c~L1!xEHD&sy"
    "2fhXJ#ASHsCtdNp$B#!__nZ;KSa+P8A)#;`RaWoo$?y=Ra5faDCwVmU3tGB50B@to^H|wxx?hjJqP3$9TINmL-"
    "7kCG^ZEHb;iVD+QIh;D_{f_kX18pSSqUoh``E)=lu517hC{vsePfhC3#Qc|sO<Ni1CO!(fjKgn9vk{9Oy94?me_yw90zLUHue}?k"
    "aZJdClAbwGVD)M{c5pgF?l^HumZy&c`)dX2)!zr!LT$~f29exW*nMQHUp!6xIM<2;WG6V{>y#8J<yN9KI?w&mK2~5Mpp1gvbdP9$"
    "EXHyfUH&LXCq_8v_BsQ2o-cWnwbE(>2PKyA8US~^tgwaO81F6G3V%cQZUnnXbkCmBtS|1$S^@R*i7RZvP}-"
    "Kd;PbUxpH?Ag1Bu)a^AqGJtywCJ6nE+&qEtvI{YMk&NSFA)^Mgthb=##J{u~}N!kQoJvJ^9>y#BZ9@}Bfge_)y@`2Jp&!M?rR!y)"
    "mvCjk(-#>>D8bT)6VR2g{)<cNF5F|@W>yE}Vz+)`&%^BHE@>Pzd%<5IGT$6HP{4%^Fvk`J-sI^Ql-"
    ">?1YFvoGb3>~B)7TLMSL6DzOi^m*d0vrOhN7X@25i}FjEcB#2(5Ph4P%`QLbf#YYIUEy&V>?_qllvpBW}s{k7oqPa_{6*|?htzXW"
    "l?Dhqh}4v$l-)`3mr`Rv_0$1zyzhc+Yy+(+OzY^C!?#w2f3m?-"
    "v&?~Ma8L(BXSk!`qY+bf7>Z<5n9+CXa=fb+7EAZcyxN0Mop{N58{+GlKudDkcvYk%DG7p?mW!GI^SFKdfc>w4{4+E=-"
    "?2WNzAYkNcSXdKjt!9f-qi1UYE-DrUltZRx*_w;j7tIF$);9!4%72!!>^wb5}M2J=L*t;tt2h(hwr;PK4KrsLotlHwp5WqC}xDw>"
    "cZ9DE#>PoAb_2$nXOT60aN2n=N+(xD1n?!fj>edwfHCJR9`!bVk7FmhEDDnf}9F&}4Vdl&P0X7C&ikLlMc=6=r=<(s(yFr)F|xSq"
    "4F6YZO0C>ulW&bYq{Z$Vo|bEvTbeReUnHy}xV2?UUIa7E#SewAnvFXrr%4laH))BnJkf6WAJ8^{TSLWDU=O(i@sTr6OevHmz*8$R"
    "~^(u-"
    "lKl;)=rWG49JlV53Nc<PVb5?30*WA_k^lpo#TQnlgCsL5vmnGlf$&KtneQ))r~%MYIM%j`awSIWv;zBU|dwYuLf^cD*PB+9>~K`^"
    "?#7`2MtOIOAa&%rFZEDxLDfpP$<Y_7$J0d&f>6&HPG}26yYX)kwfq?KgMZWeIdK`pY{{GXs*ML0bve!LI-"
    "m8}KbW8k*IZpY1wt7Fw_`haJfoT>5j*1POs<gL)?26-"
    "m_W1vI7&x&k)jlpq7bGqUD$V?~ZF4M@sNhoA2GpUmySNFeInpt>uwznCE>aS8N{Bmt2f!7-"
    "YERJWcXZ|EWN5LN{i6bhqWb(hDR<9!IhJKQDku6y)^sHajm0~5?1)8yuf@e`!rEAV_X$GHvWg4yGHhAVYu@whiGPfHs$4$_0luJb"
    "R3Tc-sp={mxC4Db^mw4NDiGgf80XG`eX9rKV#Cq)Xti+#oHC~l2-"
    "tu&?aGw=lC{b%3FBM*^K<nT2zH!5h3a;wQ=e6dMQi^+KnLdA>@Nk+4!YiS3K5Fy(yEtuza`1a=_{0&pE3yce@i}h7lZCFLmtAf~G"
    "0~#o}P)%B;05vKCAiLpuq7(Y-"
    "mdztdeXBVEu2<zX?mTE$K`G0VXSy{=6^0VCW@8MGZyF8{E6BezETF~y8Uy%VI6^Wp42@<ggCfy_6g46h?JiX?IEX2sIq7k^yT<Hf"
    "+N_z^AAE^O^|_VP7|HZO?l!D;G6Y@fMGSlGop<t9-e_uVU6#&}a33xk0)MOX0}#2@m=*xKkM<$U-"
    ")V;k^ejDztEe&mkwgzb45gb?;Qdf9KKL9fql-_)@I<N>de_thA+n|CwT?YVYhF6MD$_T)18AoHv^jw9=B!^2rHSK>hK-"
    "%TxzC)7b_=)#U)&tn0Hsiwf+{t$q0<37>}m0<XnDw37TkSMH6f#@jLlsu#zP_zir(@AS};6$t3d;-"
    "$e1U96znd8xxP$lgZX}c4u*ObMkqgv{h~eBK)l|>T@S@)+<?m*E9!1ef7t+i=Iqf6`+BKIAg9||TB+?<gC;%Y!t)9558=WC3l<vr"
    "{C9E~v<?=C;QRA(Y9@*^ciLV39Na;Kwd#;-KIx`xDg$?)hnZK-LJ7niU0PFP7JCRVV1X6l$mO%-"
    "Ac;b@kr$k_s6jen@okz9k^b<#NkZ%}6PG!!%dD?cZwqg|2Q|l+V#!p8LB!9!cHcaynL&EX1aGJ(3GCUz9pf5=gQ*v5a2g&r(Damm"
    "M1Hzi65I`|0463*c+5Q}D#<3LcTS)H+0)L<K{yw6BaN=kQ$&;ossigHf=!bzV4Ii~PnIxqUImPF3JTi5FbrzecHU`>5MHY!=0sCE"
    "nrx9`7a$lj$O{z;x?kLal-"
    "5|4Zw$uM5F4hBhioCuekpF`%7ORSoIsQY6tQ+kghGe8?J}Q7P(hb~4)L5dVG1WDe$*b*X(Gh#1Z2kiqh>n5rLi0XCE6%5Gc#5HZZ"
    "=2al9=#X1x*fIMFV0#q!ga-QD^{&vdvlDXDwa`WUx(9q+j1^ZlYDT7&r~zwy7tcF|U4beFb}9TRQ<u0Umi_zunUZn{91+Ss&x_0X"
    "*p;V!I|<p!o*B!J4eP5}zmK;pV$nAqOFk!tfEYOF`El*Pd_0?2A!m3tvp|9<*Ris!%E4(HMT_=4K|jttGMK>fuS9H2dZOVBO94pu"
    "aEPEUtw$>@TTW><&-"
    ")Mfx0Rh@n6*_AeR2r+F;Gc0(v<&w=+b65{h@yx0mYaLDf2a;Y}p_Qq6f80FeE<pMy@QYENXd0BMpLQ#nk`D)q?kOgCP+$E}+tIbp"
    "Epqj4MlbcT2IXsdQx^8xMfpbi!!_0M8lK*&%;Ln-3(Rrm{;nNY-"
    "^a{=ZWH6qwC`gY6=01R`lZ+Om>tSQt1MGW|TJib7_yzoPh3zHHIg}shllgB+^fnAlI<@S}5Jwdu6YvQH6b^W%+IDpVrpvLpxDjod"
    "Qk?2+`9n0jD9M>2nZs7;9hwk<H%t3j4r439OQZJ|od+6$=pHTorF<wiZyK{6FL8p?$0SyKp5VFM*=h_H(nj)f@-"
    "$@JITw=dxA~Sg@ns)-n%X^ddaGs|K?7lwHiFkyQh1nXhAJ@Exaq7#UI4Dj-;WxDWkE9>B4(Y1Yv(u3M#HhCc-"
    "6x#t!0s8k2&j~=dCd?aCxTo%WQ|>WPeZ^DGAN-p${4K_DI;MtryuZNMlb93wmoaN}-"
    "<@gLyIgqH~z+Po=%m{)u@q6yP5Bq?7&6f6b8$pDjnB9nC+}9|n6co*32CDSOv1Z0)dM7)j6ndJa^fBQvr;Pey>!1rZGMVS?0pP!m"
    "{WL<W?+6Hh+<>=A0hS>AMOOw*s~`-Bw)lj@iV-@|k%0w4eZSF}~XIZe~-"
    "Dc&$0)9P+Fd`@BC!aB~@1x^_>=NEa$n3e6ITWB~DQ8Y!t1`I<Jl-UeD=%VBw^cUcitd+R<vkLh0aN?7i;m)>0fWW00QN(`yj57(d"
    "zBK;2m{MV;<PK|ecDhGLqRfH6l<-"
    "V=WWqc4$c+4k{8_L8Ar&PchxZtmiAMtFXM?Ld0piY?<=_klg`N@?2w^_pJ`t@W&{=#6Vyj;J;R8_W+6?G!Fx&+g^0;-"
    "KE?G>H0+Rx!$re6{Wb-j6|NDOQNT{q-"
    "JRWeJq(G{pWfRZw(gA}U7H%6=Bukq9Su`}c1wD&A@>B2yCE$=MK10q)#xS9c7}VgThw%s#SU`d7if^F|?i>f9rgct^z8*ErbmFig"
    "&EZq#ZbQ%iP9!fL?%hn(-0Z3NghTUBFG0bjjPHhvOB}tJ@t7zMnqC-"
    "&w#)qzA)!#AS$I9nFMcEaHMXM1i5Fc8LNerDED86IUW@7SHs5sRXg{0>TFk{o;8lEX*nfN-"
    "9Juo)fTE8NI}NSJOvz+YTpMJ=7`;E1>ULL%26=l4`FXlyw@2_yBLr&<7gAFccNyp=w80p|8ZG;@!@!0W?Y%XZzBN4Z6lY<q&x3hj"
    "$QqO){z2$p@ez0Pnu)ZrqdXUYxm+FxI|@TId}L^Ert2t7Ho70E>wl09tTqr5ZfGnDYE~B(8H{s8d=QKhH+UG08~y+6Ddb>dknEkE"
    "doX(})vVKbBYRd<96;;L3rz<z06(TZ*cG5k395WPivde})sRv4*|Xj5XlZNW33CucfGm;~$dDeoi6q_Y^rHUTMTt2F_S84@fL)~7"
    "x6nk=IaA*c3m@DYq6+DdHT8G|uNi~w;gVjDkeN3B%{kwckj?Na?7yAe1)?Y{I?u1n)BpK`Bp|Jt5RCv30t8F*QUrtT1}a;;Yw$jg"
    "PJ&r21C|IUEC<+o1kFu{w_S>2raNP%)N^7ZSB7234^RN7IOz|2dAt}UMSy+a+kj}qDOMsmCfiKv&b;s2B7JI}1O|qOKg$UhEzihe"
    "PRd|(b*6{U1{|0py`D%7?3ruU>+!&f?Ak$=0^qhgRHEF0dm$ToRRbd(MRDmNQ|a^%@q&&7*^-"
    "S?v%I5iVqxtoY>fPdi@_C7PHIC}G{8xLICIKy4}~i#HO@}|$>qs*r{&Qv-Se2#G)<-"
    "E4rpoJZXF0Al=8g>enE9I{cHTj({`It!E&}BI$y0tgZxVvPvv=U5;ncU0|P!VkDk*?%wT9WCcN9F10zxslk-"
    "M|<_~&=`4_o$uxVkQ(=9-@kSCFvio{WOHaTY8vD;o)?BP?6_oQRtz1CKW07!2>azLBDrdFIB%>%cvug*vVXBB-SXhw{N0b;?631b"
    "fyBqyih8G}Lt;QhFa3F`+V^P`39@WZJRWOsqi1oOelwr6&_yQyqaXb%huWYu%<4YV&DZ5hRcD7{L12p6;x&1`-"
    "(aPjj@KdRSJeEG?6@$0F*P(QsmT!6m(I*D$Q!tATWtghn-"
    "7H0VvtzwLYRw1n)6lp&bq5WiOOobku4EQ)DDWx~pnbJwfQvq7e02pAs-"
    "1{?1q?>LxcWALi5;h%X`fy>R9O{2OgSydtI5ho_`A=@uQOxrxdZ5a&KR3<5l!0B7A3wv@7S2h>7_p>i%T|PQnUbCQKa$e3Yz2_Jo"
    ">%PwPrVl&XlF`B-IJA%nj#BI2K+Wq<Hwakxd-)-"
    "K4A$3@GH1KgWY7TJ@aNbN)#=rWFT_5N?tqZ(%7Nk;|S2Y_phWzud@r_{+YKSo3Z~@jr*a(y^C&GUU^#9&oeNw{U5D7gKSKVcS^TM"
    "XIO+-Yq%}ff8mZmeKQR5poj1ZVqr^$U=O-H1yp6+TJYDHSRZDSn}K-"
    "bI&f9fAO_H@BhIrlpwbhMD(QZk?WJ=BE6X@YPxkFa_(BagJuCyehX5C#y7Jvj%FYfMQ-IMUKj7cw(0DuswFmRNdrkDf7KciV7eSg"
    "{@)j(W;4j&er<|(UP&N<Oc}@(3f&>T^&m$Nz$x-mvBfkQ>%lZtyFFbxjMRV3%YcRa$kxwU;hfqoI4MvEKv^{pDKJ*|_=ROX-"
    "?=dU^&#6OaP2nK@cM#>lti2y&5%gzS1p4KqWXA9?F$)qu9jY`-"
    "In!e6<uN^3clBj2TSlzf3MYXBdk)hXXQ|@_lTtv7OhfK!LgA~+94v+-y#zD5-Rv9op$e-"
    "j!OaeB5vK&>kc9(Ib{517lHPLsX;xz}56G4+IiG|3jtE1g^^HWefnmQ$9SPwP++buf%JP7GP#<Qe1;~`{2p;dJ$_cYGr|EQGEDJI"
    "KS33Z&(F*T2`!fzpid=ejE*I4K0MHe@AIVc#PJHD7@Oj;M7Wl|5jcKPnwa@s|JD*#Tq=@wLIr>m9k<Nj1fwHeOIcOG>kVkts8`T%"
    "uE^O;ueDoLgpL}WjXi1IZs116R{K2vt&rdq2?FEB}R&B~^nD}6WMGxq5ECGUtlm{3JkvURCj5uI!4-"
    "3pg`Ey*<!y>oOXk>#}20j{KXa`^U9wM<o*GKT41sa2Xbestwyk)}_F9HNqIDhZ(w5CgoK9FIpUyVTe%+#PDMC0`%xrWtLYqPkgFk"
    "zp^**H=}{j_YOxrn~>LbLu3?Io<IRo4cm5q0E-"
    "q=k5_`frYL&>7inF=Yo(7l60h0wG%}mWnfr!IOk?2Fagng7tSLFIPSTd0tC+_Zd(k6Gtf+ahXD{TkT7&m@k%LoD3sJv9^-"
    "D(wKe4yl%!}jBI0aHCpaT(vd7v)6+cOqH_hbtnxj%71iLML%oM*&oOqQ+fCDacZg?5Vvu2EYtzjVcUE~F3D%rrUqfK$(81D|*+F7"
    "NrjQ&JDs9jHCObSKziTP*z}5QPoVREI$^fPZBE)XqJDLda2a87Ikv<%LsKP18y9QGx`D~6kZl2gd@_tRX&Cy3(5q*SvUrp1&6c_Z"
    "-Lvw>IhWS6{cQ{leEIeW!24wBuz`|iO)I)Uu<1-vnRv4Ef82OE-Yg8Ln8bPiooDLVQg&Tyu&JvpfLgk;*SLp4Qhn$CWV`IdGo=p8"
    "{J282ii(*5~&r*%+lGZa=9wLp%>2PKp&jW_Se%~$!&r>{4k4AE{1qef%CfHnL^B#u_tIIG&+$^v^DZxB}HC;hKym2nU9I}kKJ>%Q"
    "6IuEprR&bSV8{iS1@8|A^^bET*8oUv!4YujY7H$Ktc#PmC*B&RW{ro`Pf*~^8lMIYwMS!(>3lKJawTP+gN4%8sAVA<^BsWc;5%HV"
    "%zy%F}3-0q%CD4*%89&n*lU7Dbt73<O<V$>=me^^Y*6Qt4qX6ipH78*`oA2Fr5v%}otdUHMy-lU1CpI&=^{n(^l_6}Uo&NkpZ(46"
    "q%UdZfKLpJTVrAczsQh}cI3!fKJOLHB3+(Ax2m*hGYPUrP3kWcsh`Cz+VVEWidlZzImW@C%vE6~31;0UQOJosPXMWbUTqL|ch;Vl"
    "{^(wr-nIe497q>I0Mshj_Gpeu}=By(G<<F4!kUVU+t21GyX9DS$EUee9$tV$%6A9K?@O0*uLK{R-"
    "xYAU}6cUCX;A7az1`<K)!O(xqQsoTBPkJ)nXJ6suZqu*9Y7$@7$Zr=kGwY+qs`2AExMBD2>E7ozaAsP6W$nuwX>&x$NqYvt<H*U)"
    "2zFY`jwc_=8^CF}NANPED*v>*o5u)9F720Yg1D04$_&$}$nhq0KeOk6<b!VGS{=IX@POcWBFg-WaCO(@TlO7NX}}bostO;B?v*mp"
    "Q_LfRii7?whAZ{8nGl0@Ml*W&(}Q~S#WF{*Ts}QD9v3)90__@CVd?Hfl)?i<&**x@iff#mjgx2B7aLc5h;R)U+lT^iE2d9gdNi!c"
    "JyQz?aRf=veYgF=NFu2RQes64ruhP`p%JSE?=h-85?y>;H`ZAU&}yT+(2nWGWRl2v3IM{21IGY!GUHmKZMH1o|1ild5-"
    "T%c4kPM*lYkmQ>M=<h%}FxoZ4HlRl@?geuXW<_cK}JIC9I3j)IO0OdM*ntSy*Mv^o|7XX0!y;G5X5nH>Wo}28`j*l9dpzZrHaV85"
    "Ncq+QhcG)*1bW<1uZPc^xNWr^kr`J3Nnx+T~5cKV=h<GJolf1tG1G(+}r=r)G_Id!sQYF+G4m{!#JRVEpGdjh6xRAq%vVG{uub#t"
    "6f^zZ=Sn4^?j-e}jfTz{PGaS!8Ei2tR@}?s8`KB(@CUz!NR$(DdpG)6SrIY|Z%msaO$@kxp7LlAc-"
    "oE3RMoh-E<xj0vOz^i1-Zc&8_bU}J;#NB_-Mu&*zC+*?LD9$bbYElx`>Eh;@r{Mn|A<DQt|jNvkz_MXdpn1|-"
    "+@UYg$D@5QR7(mFuOt_xcIN%O_jKQ@%OD$&eqtDalqu+)SeV|Z*TiaFvvk({&w)5^Yue4O_Go9A)tCCGzArHKPhIlu>yIDNmJH1#"
    "#5U1HjK>YCF;%jckR7|8ow&KoJeug|b4LgCo&&LfkfQVLd*hw*LSDrNXh(I&q$O4C~U>4g%l<$oDMYapBM~2y89!u?(V}RlGIxqE"
    "*0@u02L*BoTZpH7TC2qfM`^Ds|0FVZsvE9eX`MS$8Be+UTXiAr8?7UDupvhuDR1yTZJkYbj4a+gCp5=jl624zcT9!@5*0ae^O2%o"
    "JLuLfZ|4o2J@`MjBc>~Zy2{7449-"
    "uhNBd*0gN{a@BC_e|r9Em7W!I2&<w!)@p$05$@&)^zCjl%+?GYYNli)&pLpN9!~8KMB%4(lI!%pq){B*0wKnM|4349_eWF^}($R8"
    "4qd%79<kZ@H{Zs}$K*HRj9nj4cuLrpZ-;i-|AS!2MxFB|D;)0856zHtW6ti2$eXQ02;LihPU_-sGNjhtWh}w#q}g6tF!`E=EMriK"
    "f2;v{=gKX;-Y7-"
    "pSy#_Stq`mZb|ry$vW3lJhPYXk0x!8B*{|Qw})DC>GiCyBglq1mFY=_naw0qpq{>0YN=;Lhy_MQUN!RS}U|&p22DP8Zx|~oPnrgo"
    "yi50kT)ZaXiOg4KE}`mk3Xh^j@t;-"
    "g~W6{OlI3UGr?M$B7a=Sj3cT$qv4y$AUhmRGMF@mEnuZZ7P*sr+q<1smP_iN6nEho#B36nteMDwd-"
    "B}Cfkn21fpjqoJv88V(nokNupF2h5oM{(!;ZZ)6_{5NlublA!vi8-{Yu;(EaU*0-"
    "Nj*IKY<)t>W~^+dXDrXXH4DgsfHjSbe|Ht^Inh1Fq&ttj$YHO0mdnxJqWH~VZ$c)n|SPOd~=G`1!)^$c8zrd*C#5w1pYGOl6x?lb"
    "6Vt?FO3s&?XS)p_>}_#X@hO{LlVH7?fHnrz<f!uSz$hl2`0WQ!N~*DVnQu0sCiBGY3C^(vGP{H2TtyWPx;c61?X?P%j{VP!HDA8)"
    ";)?)Y|WDVVRT|Q@@>vf%A_4=>8>Ec>_y+TE9rJLYB?aCI`1$`5|(9T90*4l{Jj}&k{ophCLAq4kK>7AfsQK&-"
    "@hUwHPhN3I}ZWEN|1$4GP*z}mvM5FRfj*w#H=OKArA{=I5b<@)4zijLbg9~M<giLjf>`)rP%aLM&KG2J$tAoJ)pdW53;>I?XBNyh"
    "9B+8u-%|zm(H}-GI)BIagaa$I{pcVC#K$ziffbqSYLd4^RVw{R3~L;_S!|G5M@W+k;5}#h@gxo$%sYmbj=Gu-"
    "yZdxU9C+KB58kL`Df4^urP;NW93!<@RZyzqG6_IYO?K1%_UFP5quY@vlL5=hn~6D`SlpXZWo(1MUyTGd#u&1^5gPjr-cPeixW+E;"
    "1Pz+)zA5L_`g06)Q0|*=4jjhnCjee;+%l7sQ7_x;0vbv4H9bV;OFV7`7Pme*~!BFn{fDK>V+v{He*SpBgAln$|lR?z#s@8DCTLm2"
    "8IXq!3ue{a=LGA9L)_|^SNnaMs-NuiI%k&*@~Wt-BQ_|KaYPXL++S%&qSZubf&EhPnLwtXPO-"
    "Z*@Z9o)kgg6U3{=UG><Wsx7*JW;Tp#we7V4n6^wvUEGI4wx65VV;x8vZAqb<Na6u7b#2S@*u8{|kCgBCvBMc#wMSZ0d3IrH}FUq-"
    "}v|(eI29F6LS!8tBcS9nDMf0*@c9=)%)+H~Zh9Y98Y5zeqGX*u`(eA#zM>J=s_V!)YegG{eZNPTOhGc`|-"
    "e_VDL=2dDgfeT}ahB|Dd8}t#a2U}T`1Xv!{A@H~P|cv0qK?l~%D56Vcj)y;`TG-jh;hgjM){tzCCuv!)=y2Lqu);=%t0$FM+?jBV"
    "dx%crjD!~NGZ8%zhS9JhC8?07{hJg9E{nv?~2)zoWVRrAZ~j}*Dw-&clV`nTz%PKZ$uq}(gGCw&0l0j7{nRGgS|$FJV0?;!0;=Q-"
    "UD5OPT}-axLr(UE;V8ilQJUf^u%FQtF7qG95<)IY@swIj(HoTR4hgd>jH+*L7}(D|IUx5FXvgG&-"
    "0gF20;~T+h(y4401e`XRTb@o{XE27`<A;I*QekTuwp*Jl7YN7d1GL9s&R?Up<E-"
    "lJJHBWxLfF3sJy9|Hc*zM|MwO8G!KQJBgHb4~yF_fDrbVwMDlTwbFHX@-"
    "c#_B||5$g@_PfB+rv;Ny3vNjOk<ZM0z9`Nyc1uO8?x3gxR@s>h|*h6nqC}jfQ1k24i%a#ufb=GM#I)-@g-"
    "R)A91;T@R6KHI!^x#m($)^Hl#xvboYT{<kN&NA3qVmSlF?*?43Ueo=OtPm-OP?vm4{b3mLuy<QKF{!kl#Ixx8QP9v##)4z*($+$T"
    "{1qTw$ix16%Eo~qLmfbM5H_Z1#Q4tagQ&0>E%!>6a5Zh8$6>Yn~)-"
    "g}eeLxG@Qyh{v_&gZIqeyPmXL!Oe3<mqm`2~_gFO`t0b*5dPeR{1d7%@}kzn@8Z!xWS}W?3JcerD9u&BntCgBnG^V}}>qzO3@ipg"
    "Fd<w3r~mBjN(+;&jQPrMxf^LTZZnc#2f7CyZ&oAuOmN$fWwjvF2t$Pk&uwSl!YitYy*&(KeLuf_f{LCG1iCEay+}a8F@e7+sL$zE"
    "KJI@GgT)R{fBRCI^ZmwMBVZamnM!jI8Pl%Xjn#NQr2_QBq_>g3V}`TxLc6E@!NZq1KvllE?~kE3lPd09dl_7xo@-"
    "+b}M(A%yb<t}NfyGw>2%RsQyXd7N|tK)`^6oHn03<6(u#6LWjpDz6-"
    "ApnCG;9MQC8HK!1dTNd)w6Vivbn*c4VRGg5gldl}pR!i>&x`b1L0p3%tawzh+F=44hsCEL*6QM6je>47$yaZhwfg>o+bcQ;q6rMC"
    "iDjZhhEhNi;pP9-"
    "5$5_t##pEcH@X_b+YG_iXljgFu6!&Q_Uyuj5fVwdz^aZO7H%CYnC>wGySfXY9PlqK^ZD|<fK&DW!9D~ad#=m#kuPh$P6N69?z^cv"
    "R4P3(K8Bl4gat#n8n&OuCS^ApVBQ6u1!<g-GlHK6;CB0EDS@5H{KS-sJzJhb%W7nepi#HJ>Ni)gQ=6RZy<Z)H=D@WF5j2h#@+lx*"
    "DYt#7$y!!NCu}1t*)flq=BW5r@e}7syJ+DZoL!2h|R#J8rhEv_K{<_&7P6U@^6Q`y1Co77%;-<5-"
    "ThV~?4I_*G<}Nh}#U|DjaMMYa(rl6Gjid^ULtaIWuw$q5eOvO!HE~RL0Zq*LEBI-"
    "7HV0t{I$>HKzyoA1HFG+D6WL#MYKpxdLqR5H9`+E;#PJKaWyB!RGtT!DYC|VFHe@=#aoW5Y_CmuFE$JTK{s0&6S;X-Z>}-"
    "#d#R7e{jN#0vARwVCuc`t8HU0|lGjD>ew*KZZ*Y|TqhH>+({T=Rg^%{*vW<}n*MoMWoRUqLgNk<Ze!4tvB?BiEmQ;>Dkqrhyt@-"
    "!Hc&u9q~exm)zi0v<e0{o{))Ie=%;1JB|`3Ho<7`Vkj<y6-"
    "_Wj9EaazvYu@Hgu3H5g_Hc_;TjNi6Ue*J=+`6p|(&)pWY?itE24ye;fxs}uwD!f+Mm(O2esi}D@M?q&y~04cgB#Mr)P6tiSnH1^r"
    "Zen$5hUs~jbr1xc|iSmQFL6~p5pA>#f_|!iKtrsmkJZY(rxjI033mr#$M~h;u-"
    "v}wYQA0YQ$iG1gMHNFT2emtNh<0zLivx^&Grer)P!fY}UY^+WW-ur2yQYf7E417gh?k5$;UA{myD4x@<cYCD9!~q&KjO@62tw1g("
    "hL`#%rXXz9O(3@-GWO)V+nl>=<E!V(zAy+ZS?J2Ecmkc!eBkh4Cfr??tX`%%paJv46-xn61CO5WyXc2=(Sm=4TghQ8=MLF$gfg-"
    "Y8um2U`3sqG_FanpOy?W?K2_v@OqRrG1DL&%eEa$(?K?BEev>goy---1t4t<?7uNhB2Uy4mo{eZ@?y#ti3vIGPo+5h=gdjpi+D`a"
    "xQ5l3sK^7MugFM4h$0=rc3{>t3EKS3;vs3x+|H@VB5`JfTtq)$2C6Vfb!j}oo`6mX%#5C){~W_EL|*8REPu70C<4q5x1mrV{18#m"
    "g%Ct*{xa|)h%JK0S&<XZ#)T2GYyVKqVDrxJ1j#$ACaxksLnkwXE_4hRq|4C^x)4P)SxBf4qbh<KWEby=-"
    "9e4WfgOh33|#~MPb)qR+C!Yg3Einw@hk?*Pay8Bm80Qc;~T}IYX2jr<|3$pq7NA@kCKVp%MFK@)favM=pL|0O|x(;#cl8J;pQ`iw"
    "g(R$>-"
    "&36u2qb}2=&s8YAt7tnY6Q*RP<t2qvtGM`#H^)f;77v!KQ`Vbh8TnJfp&p4Wz$S9a<ZD({?n`Dh;=3WM4PUFHP1O&3Vw+SwIsm{G"
    "UIYE};kejF<H?^*rViZUh7un@}5moYoY#3gehv+dQzJAC7vB6@Dforg|2q4*1KSr&nVR=<hJqh2CHs&xW92c)Gj0zUgywuQhWm1a"
    "N6G=()<#MHoNcJ#GO5c-"
    "14kL;ngEk8?dB#j0)>@180du|NhENYxMrU>uI@1X<Q+nh6#YBVG4*c+k*hh#$j^i3}@9h%HicM9I%XbGkctX%z#j82gX)=SWdz2O"
    ">7c{bQC`6;?8Mm^~4K?A*}&a(C^&Q@(j{FjZQ!wrG_koR$ShE#6ZtqaW6!&(ll!!JVgpwYJf9f8`s-"
    "7AH24f~}P|hIRSja=TJ%O?vp~8L(2`KG<uz(j2Dfr;hKla0t^BYW(y9#}t~Fw=98f0g%~;)>>uSazW2w^cjXa54<TRqvKnK(<6@M"
    "Id2utG3dPwGFy(=z}1cf&J53$v!cJiZVvq7SB*FF=j$?kcsTF*<7etlJRq}eC3E%64>~LhUrCp2kR)bMT`_;osj|QlHqqg>E6$;<"
    "=rPF0t|)K>03$Ht2|kchTDS|5kOQ;KY?U%i0s|dS{^9uMQSu-"
    "b{2pSOc^+fd!vi~P1sHU<aRWZYqM}$pC=zZ|H<%dJh%Ah67*{~fn>O?KdohFj?;Mx0WFW39y_&X>N`W{?>o`=5Gw&}fVy7onpx4M"
    "Vau^fk)<t^g0c;=4@hr4YuQUQg$HKuVNR0gWY~@KwX)ocjaOu9#TFl8M-Ob(-"
    "Ch$=BKi~6pIn3v|4<3jY#T70S)<yiXV!BMt7n61R;kC<VGZS`02+5)p7aZ0I7w+odbWe)L?TuG+(x?|Cj9i7bMGn1V9y&@?1G+J?"
    "nGPT>1#pd%;)zNOCX?N3dye{psEnP~vsTWd*dc0wBw;f1?OZQheUMHAyzZ*GPb@5!b*1mL7Ke@6gN}G=PJr&2;fx0P_DZ_!<#f4i"
    "yHo#8zxkS_V3*(I_YF6QyF2ACAVQ3zNUyMh4<ZscJPvByvB3I?06=Tx*zON~Shn+r$IJ}`xDh;-"
    "!DQD(OwV1FEa=9BDZ<RQ*^9tvRV!e4ue=rwg1z7lq>zze%F}Bhh0%6El#vr+8JV?gk1(q*NNNZ%X^>GCRgybzJBl!8Au>EVd4`QX"
    "6OwNx!OQy&{zwe<WSW_2d^U#BOG!7BYk#_BZS~3!f)>Q2d<ut<#sJ}<1NTWz^0tBtqU!Bl=}Gc7xgy)*z@My%ANUXRL)T;?`v73C"
    "fSv%rG)+Ybm8fNJlIhYAoT3e)E#0!fem&cj>;WJn5l7=+69yi9aI~!DWJ}+Uzd1ccjBso{jSZ#Q-@)79W6P#1gwzClW(-"
    "EAyF3%f#G)9A3e!E%7uN7!k4nLe<(TCdF#uTDvu!*w)aME65bXvaSJI-?h~u_S-fBwo%z1?bdvZE#g<J7cp}uoynJ5)-"
    "28<%uBCK%BIoH-)UFMxYc}jtD1|&AMk*!=}2pr3l>=-"
    "0Y(sj)R_OyniC?h9CJlO1in_xX(&N<(y*~rgUdEk(Qu~XzpXk?x%!~DHWZIJ>Xq_VDXBrA}aDB-"
    "n;8`#M*o&F~E4=H6v765~1c;5oIG}A%=#k&|e966_TV2^|#(gvED_Ysn8QJ$hl=#&j3MDoB@rH#iDAG5)@2$SW=paM@(FxBxEWBU"
    "NQFGjEZ2Gw!7qZdJ9o0E{}CpMhzNUL}kUQCO+mn{u$@ao*tK|8uVxFFU7FMQ~!5Bg7RX|85!lKwoxu2(H;`j{UovnP0TH#*|qsR9"
    "F`KhF&^*X_L)SPiyz$FUoNYT$bwN_afd=UC+95CrREG0zm;-Yk<-w%POW;SnBIrb-"
    "{X&T%p2z(8DfKozxdbKajXGl)G#vchWKB4(uhkjH15#gnoT9v2O)lRCie!Fl{2WD`;HGA7g0j(kbC5a(Unn|YGxw`%+hEGkMq#=M"
    "u5%GQ<$AnBB~Vx^(|#f@xZCm3y2#Bs~^VoY6%C9@b56)i<wDm6*(xJf3O^rEi9Q(JI^;fVLhl|6;rmRML~k8t?nzd<zf`5wF1z!k"
    "6yrVxAVk4bxGXJA!D9vq(xM-6kXZ865Ek)zO{5%CP>&y-gurx;Tgo0zBJ0$@3+1^g&aa-cnB4pP;$AxjjVo)fbzqQ1%L?soWK=r2"
    "OW@E3#$VKyB1?8tG}Ge*yu;2udjVlW2TcpPY2xO)5}BgyFrL^ed@gu}*gpL958o(>f?q{kfg>@E)33^SzT1E^dnGo}%^okIsnXYg"
    "0|(5SS=19=1LC4se#If7@U4tNh)Ft^Gpa5?Dl@=l5>KwY~vXV_(=^)$u;pFIvdEi_G;Og6;xjG8}7ALh7uScVqsSNZ@-"
    "c<kk~hj7L=TRB1OhAL8H6+J8pqB){~D$mfX<K9O%tIjSASGSXXE<Mn2p%e`}J^j)YT6qt{wr8lQc&^Wy7Y>qg^fIL8#`jcp@V=Ky"
    "d-y5TIWct}PMBwe+-2o(iDR*|Lq%uteW*e`UkgI38{gg|58rtf635>}k>GK-Afcz86O5VT10G01dg4u6qCk^&+4C(a+IS!uGrab#"
    "O>rA^(PKHs*cq{?N@K7NK4<I8Ij~R<$gTiM_N{z)8bH9D1|ePS0nc-"
    "5_B+N_sR*3h9W*Z#tX~Y7^u*22jeRJVnVAN$VOoS}H@dXEwOww^=G%E|!e7Vxsag1169F4y_3M!otq6Efv0y|%av5d%hAe)&I6z|"
    "gBRe;ESybMuhJb+Gz^6Jf=&;vW#Euxl!fX9q-"
    "ZXU4@w4{akf^ttN4Qg+avID13@#usiQj6M{fu82ATqCjO1{^a)rV&Mnz8p%4(MsGO8=%cHLJRXuHDD>Xy#T#+Yu&jh=IR7e|!fx_"
    "5>VMNQseQAdM_`xX+aB7^}D48MGRxn2um}$igQak+NS{wj}iIuJTQ^51AWb<JDjTYE}#)2uOP4IUta~-JYB095$slWM-"
    ")1Y)c(uErEOJz#-O}DRkj{au3;SG=zWl7GbIYU7nvI{DANYG`PMZ@-"
    "Hrj5C+UNd4T!Rab&p$>C!O|A~0TmM5!MmYTHY*d!j<>H?Nx{aOz^bdL&vCbk1i3?N4fJ`Cj0f^wrB44Lluj!sj{8)o1ao{V{+GFQ"
    "r$yn@J>AZgT!K7qv)T5pg3>&7DE61qJGQGHN-nK_v4ac7U;bPE-"
    "b%`!8$;&aK#f9i)zFEi+x}2?&n1$awMxCroiV4BL|(|ETr`ZiMT@$u`HTlJniL9fziz5Z^rkr~u2H%4WztJi`~F*vj6eU3M7!4gZ"
    "@mIzeW<CGiFM$Rd;A@HykgMlwrpm|v7~u>W)wWG9zJvLOOg;~Zqc>BYXMn>aM1-"
    "kdSm%|7{~G<<k#p>6VXt1q)p@P=lV1OdKqc@v5l5dkxP)UaTRux(5)$H8z#v-?34DRpM0xis26S__=HnCfJr*e`n^<yU-"
    "$ZeGWBVSV?{DR;RIgrCw}bxTmZ(-"
    "sA#uZdTnT8Iuo&M+%Z=7z5*B%%=hzY3MlQ+V)ZSlFOvQpOxYhT_?H@uZ9kJ)P$!+2J*XAuZ@3qnVu<D@}%uB^pfV8a(i<7-"
    "jUTCoDx%C5(>{@uKHRpyKnGanze=skGOFOyUWofXg7kbrVm1ExMfcXR1QYfG>NuaJ5Gc&Pa6Vu++pkz6P@3@h8B9yeVKa?F2(#Lq"
    ";uI+5bORcakJa)+A?kFF6kUq<vFc0p+ANQ3DY6UV2zW%mXMQ6p)b-@41_)-"
    "YX6B$H^5@I@vT&a8|?__PG%^PTUjkS$|gBGXY&<6wi$52Dm3$%t}YwuMiprn1&V*(IjafNy(d6p5i9shk?aKO6Vm&>4vUW9G7B7|"
    "GM78*jGCPz$G5NkSl!C37T}^^l}f8!FQp`U}+dVis8aniYK%B?2(Q+3fb*eYKnF);kA8FZNSLyz~<3WjwidWU$``h(wVwX9#DfIl"
    "d`7s^4&kbP|jqskuSC{hMn%sj7p_>{K3bCz4KoG$S`O2eNh3u19-%AHpm3vFq*FZ;HB9K1^QTu_Pr-"
    "(L$!2e@Yz4Ol`BkVIjL(%4S-=e*qEmwdM3#|np3uW(h;e!Di)H`eQib;Goka$tC;7SGQu1Ha-"
    "A}@v}I}(@!Bq=w^&}dd}?hANUrt#d6C|@hGD$(+pWn-W{2?Nx7uZ|1}#Zky?FA-"
    "Bu?ecJCODw{!qK#(>`Qrl9RiGlLD4#s%#K;F!?L^y0--l#y@BkU(>KBP_dugJ7Z#ZR68a8f_Bmf9cChLC#tjCZE-DKJw`{K;?|Z8"
    "1nrhC$T%ZL&rDexuFvwYWU_&-P^-"
    "E7u4qSJ?o^ic^9V&uSbIfKoCXMXmn4GAFC90da##U(TaDEdj^=%(E9tw}0giz~8oXC9IN#HI#)Yn=n~i`1AEMMerTrSvJ_%A6dbG"
    "6Ci^0*|F4dT5E61NH`%)d?KIzU&0v^F(;kTChi=I81_IgLl$#)3e39Vt#6sp`jdhwyuprNN$Ih?@F?>i&^dFTA7ps(|;xVo~~zz{"
    "h+8H@TZ=A7bs5!+EyWAC5D8Wu?w++N>YH4EZY=)jS{`q7tZbBu}mkh$K_B`f6Wn&zPJU?jm$0bflHwvw;!q^R|Y(=i)8`>Q(5fUQ"
    "zf9ne|PZ;gh(51+V9AJY>t98=0+7uMv|(@s+hTYRgO+AtvV8~RDc_o%Hsx3LBpG(US!C(ravcpz596*jVwkDyLIRhso*_1#?>FQ>"
    "i#sYDi$jAY05j1O{2o0S2T>D)csSS4mmJCL_yLTBha5+Zu69e$AH(j}1^zTy@!xqEQN)2h9bZiw7x@0Wm-TBgJ!yGC-xLDyDB+8`"
    ">&_ne@pHEkg5*cl*ga|qJwuQ6ubt<*6{xm>~k4)r5SlK^4K2%?MDE4mo}nf+Q#Wr<oZ66zXJ*Y%acWY1uYB&;Sb^re~F=>$6obB)"
    "pMmL`1AA5S(|_@u$l2oswHBoauyy;pZHcjy{4Njj&|XZp=dAn-"
    "!g4mSLX$;br~Zm|5uqc)F9f;iXrPHItKK6u87Cu=9<^Csc3;t4wHIe8dLu6FI#d5dM7g=3wGCrh6k9pu0oW}(nnWF&!iZ^BJJKy)"
    "44$TB#oBY|H~{yI1MV6HGOdc}VCEs*@zJBL`f1kExk3TM9Rmbxh-L(SSEQg3z0k+2gTpC+K>?KBoaWl+~`qrT&z-"
    "$OYUHeN|POR_RA4%e&Pe2F;t{@r+=w^51$&<2+bnb~GiP4lbUfF19`zym@@YY^na>-JFrPRta!-"
    "sGqi!18fSKR>CQVVJ0Kg23K!K?tt-"
    "J7BKrBHs<bkmH^RRO%Pp;un;WZv9xtt_5%n|Gzt{eUY+zS=Rlm^DizLFYDn*8pd4r6*7uvdMWSYZ@(X%)gD;bI>|q{geU=2<%7!s"
    "EpmD469-!!i?KFBFh*T!=*4O-2-"
    "XfNb|nV6nCWd(?o0c*B(&jqCqbCGK*sW!Y~gDV%%NlRa3eWSWQ3oIM;OtzR}WXxPjNn+)my_O*E^RYURUQA7=$lH&JZp*pKh8Pw}"
    "CxD?Muin&j`BK+Esg{XsNz04F#ho6@=7lfG&RXz9#oFN*%t-"
    "5Q<WHn=Hj`=S=UIoPUSEmqt4IVDiU~kN`WeS&4H7V;T5+G1%nv0q7ioD)u@7sPhS3KzlO@rqWL%zB0Vr+L3ih0<w?irSJYd2HqpM"
    "y7e3%=*pC5Tm9yl?R!n%U;v5CeGZ%3QkZ_{4g~zF?hKEQ)EAO%xdySuD-nUa>tWc%kYQ9mMqUGqL%j-"
    "8`S}rLiUF0eTmo}?AA4E$-T5%v$#yu*SUI#8+fieVgh5dgnz)TCfcB3Q`WCZZKhK)<ZjOi-53v}acbWl@Xp7w|J#|%C-"
    "3jqJMz0=lDslH2>wn#oY{8U=*pfel^d2BOb1F8~qys`jNv9&=Xa?wgzqy#OpYCx9SpV*kPp`q*m8HdT?3c1W8QAohoDU*hy-"
    "3Tzzjs;xmo$|h#wP@EKqb)fY1Ih#Q2_QMZ&WP~QkOq(VshU}@7@zT0jJULZRP+=NM5N9yPFLGD`_d~m|Zhxy3&x6qBIH2xWH`OE%"
    "B>wBl@-=VjCHykHGb(J`WR*UiG7PstVJDcKnnF$eqJ@);kLw56Q-kMO?xzei71D;n9#t6HDO1gYR6yi25F+%ezyw6F58rCEX22-"
    "z`WfKSvefI&4ueR0MBlHxL*wMJ)*_KgRRz12?H;2;4hn*`;hwn81XeheLC~7`IJ&Lj)V_u`Z(;stF;_XJ?48G!~27HB#*>XtqHu+"
    "zg|BCx_fU!CUd?$2;_Nyowx_Xe&%{-"
    "sNZN3Z3k+OaQO>B{$oeZj7mK%{`+(f5QwR2tDSyH~e_EIPQA$k?_PVej1jvTE+1DXo8Gn#y9i?@Q~s#i2MH5_Vk4|yDUE=c}|IKe"
    "e`Ccm5Yxxv~q*?HpGOk<f3y^994xmr{Wb+T(pnwjCvcv#QbOCoJ2xA=>k}`o31bZLg;{wS~^QjN?>j^EqiD%$50aAgjv!h{{q5@r"
    "C32tFhZKox5m2heF>2yJ%h}l>=MOHyg7DrJ)hmE4Hg*bIVN;4q%AD3UvP05t7zrDb#1Oqxg&r?DcokWd|-"
    "BgGGX3hSSNUVNpo2Sb{o4Xu?YT2V8M0Z8!3g`W#B)TkYb5DgtzAYF(&<?Ys7=kIWa>E@j3XPmB&~C<q)m0sW}2h^+pxNB^5+RWfv"
    "Lhn64}V^xW-r%U`bq;(EH9)W4=%vQ5G`U8^HbY#;ikU-Rs5&*aGbLomd8-`6kD-FkM9in4o;qAX#Os<Q8iOv5{n{rCv2`wiQ}Sl*"
    "H}+dmQOhin|1x}##d3OlT*bb^i-"
    "@o{wVHaxY#ZhNo!bJAB>GtM5FDq2oZWo*e`ju^Z1lA+y%M!m5uo>>AYtQ+rTq7u+NFv0A#SOQkjJY1VSAT$Cu16#4L`XY1rT?xUU"
    "pd97jnzSD{&FCrW5lh&5f@$2tYqQ><xrlU4O@_v+TD!AD^Z%?8-"
    "LZ4$SFk(nS+&8n8<2^VF{?X4vgl2PPqWr_=|&!;Q^q>^(KhspCK)(TdwUIz?BCPYem7PA%wFi>y==k6w#AC+@p5lS3+P}P;y)Q<x"
    "GzO2A&Hheid?sMI&lv70O<@H%acSXe?~N4^;5}4xt-l^hw3Er$u%J?sMEg7tBZNhaLI<(iDykdxI)S?)1JDu&}`v=kWuxCVq#dq-"
    "tS;)63n9{uvn@@17?~f#5s!3iAJQ*^`)30wpz&LMQy@B{1-K0z+AadB+pq+BG-"
    "tDF^D}2dEn>0a?@@p;o3H<X-rmbv<RhWaYYLeHsPIjky*l)-(vlkK7ei|n$d#vmSFz-rp#UJA&N=@!?V^heCyz3!%N`WC4q)Lmar"
    ">+GtNC)RMc@%x6O?^F${v1)*G~=zy%o5d>wR7DzeL<a+Y@SOWg@Dxwd}0!@(d|I@cdO5aom&4sm3k*v8~X>4qHG4#<YrJQ=!*Jun"
    "yPGH3+{P9MF_!A6<mfuZv^2kFdd>V_OU`p=m}ZT1w2Rara3Z_QFOxH)8t{zO<%JOpz0wsnNKOOn6|UzK=U6RM8_ggM7D=IG|w*PV"
    "T(w&|Kr|IUH~)7#X0p6b$i^z!;bqfyocyZilhf%FtS@XVNC3lwARur0gVH%s$v5X<UP?5{WMu%|;^h*#OhdGVuAD{m7QwdCwx#MP"
    "rHwKL&qpm#lAv!Osy`pesh*7C}15g%q=Dy1@-T6DX8&6x@x1?OUa8-tVNqT-sFozy^$2^S5>3}Go+xW<9d{1w!$&+TD@gN0j1*Hk"
    "lEdgnKda`MjS-"
    "7+EtkcR?qJIHF#&TD$}RzJNrAY%7`lW6@!>}&F|h#H>|dV5Ny4+^8x8nnf?p=;R)P*rWm4`=9P<0xBu!ncM@gE?cYxgI~u)gFC7N"
    "t?UuF{F8(EPr^VnhPR?fkpF8?r-nmOs~E_)#<0i`tg^uy)+12JF+5VQok)4-&cJ!NL$zt@25M1XQ~t|V3$ljro-QjGJ*2hK-"
    "0Fm92%Hy)UP8pTj;E%RCYp@CFnHHD+RH@-"
    "#4dARr)+Qca}Rjwu^zwe)L{}_v|B32+50FPjg&ycq&PQIym6c<gGM``tEYC*SVlbrnExN#$#-"
    "Q@Mfx@KGa4PL_?!|mX*e?w>CrGEcGnXc)7@!@r3!n`dj|&J@JChd6MGLYHQte?JuptD^b$5WsvmpuD#pD%gCs^?&v`r)KzfkbTFQ"
    "cL^>?%PS9kTsJ8}0+?A&mX50EUvxI`l8La{3LFu0-_%6Zs92WKQg#Yd8<EgCo<O+)@z&)kgGDp6q#$MMu&`%Fd-NWBaZM2<3%s=v"
    "dJ`peVY2v~+=8I#0?}$}<5LY}+&^czTHKj#5c0KKcSclq!!k3uXmUKP|9-v`7n+zaLWzX(}^`#n)+fkb~e*LTCove3S``9^3-"
    "Awl9On1?)=MO8H_wdLnx^&BYqXvWblI3I#At;kmMy%m^H6D@6GYW--"
    "RldW%blF|m`xQ7M^nH6Ks)%LO2owhh#**O&1+lHw?9;bGvNVY<ZT<#ozfOu)G9GO;laDuqYHHmu^B{KK-jNu*ae$MUR>0ZK4rrB-"
    "1i21(PIM@2#VrrE=Eo$qv}KHJH8htY9`hz!&Q@#EHM}S*T<fE4hW&An2Nj_RZ}5zY!j$a$1$&?zhB|nW8Rgm3k-"
    "$@q^sUX0P=r6>crt1<x0PL{QgBE!8d(xfrW*A7ciW7xq1i?E%0*naLnPW1ahO6s?;GLpy9Ebtr#Yri3rr=&^Y-#!iBk-Pzu6C(wm"
    "n!BM#p_p7C|DeWIH(uYH@zXI}3iX|0TL8vA4*ezW>t_Kr^?<)m`Af(fFRsojp?K&H%R4<?Dm*-"
    "Fw#@*Xon#QNn~7|5)cJh_KA=cD_X!B8|d)`mKR%VpS3@2r1qfGn9EAhdV&u&gh}4Si{lSiF3+yKm->!F(_!Aqjw5!XWqK?cn$6-"
    "Ibgbow^6Zo1jb>tX_&f;50_WwsOO@hq0Nk=r#kvL>qR~*`S(_@OIW>iZu$Anz(1Z*B?zi9pM#LTMp}1B%S#-"
    "N!AyfB&A(&NuB~(?r%S4;D{p1LX*J(4G=<<ODF0%gcK9X*?DYEi!1F+&c0#Yu3;;FmPgIHKB5Mf?q9%eSx25k2-"
    "+8_7SKy2zBjdL;WGsn556tYiP~_|rx7MTIIVF&YdrVr3{Ji5^-"
    "2U%h?36VkO|lK>+AB&g`x%D6E2y|vnzTACuj7X}^ub0d0pwT<Af;1+fwEgF{LGPoV5Te$x>+l;^yZp4hmI9<wPzq{vh{)yDhT@EI"
    "5<1~Wr-"
    "~Z@!3Sy^W1xH3TX)n*p#J&JX9pH)Fj?&?L3GhgTp)1oemXpR%9;B=h3_i9LptpFq^7Amc6fmqVC$v1~mj#&kCr7e}5u=W6|@7VzO"
    "~k3+*{#9y)0!6GD9II{#3xTlrv*yxJYq7>~31re-=%tDVW#_2<%q%Qwx%&3`C#sS(mC*MsT3D~)+lC)FB$#5#nVxE|=U-"
    "AEj|Ra0is=&vzV9O6NOb&6Ywn0L=+pPySH#XLK(l{-13@u&Qpb1Kg*>0el?(_bcyTQ$UA+`KoP-"
    "5Hp?hiRgHN~wa>SLn%2yKx7iMZ#P!_mWK}w@e=BZiR>v_vpTkj#2!corHZu?;#CT|NMsUlydW$c^$8(T?dWhGtlucW^O*6WQ#VSM"
    "v+lgP}|d1bE?z<&9sPP1l|)(VMQPgyeS$)gG#21y7$?OxEiuYmg!avo8CLvn+&v%Yl5MLeEKl3DUUI-"
    "6Bs<{Vi|X;kc_96<Xri54iSF~EQnQx*R39|Eq>1{X8#&<$5AHd+8J?l;k8`3Jq0QIR0rC2gpLK|HDQr~QbC+0mG=DgJE?$;kFO-P"
    "m%JoD61jTZDP2)rVYpNE4H)RoFiAfU;A2k{>>q?3WKOgWzm%hM3*=<Nm@|lXoZll)C|~1u1O@SLfZ4o7!g-"
    "ki(SaT9^tL3~r2%LXu45u-5)_PO_XFj@lbpdj_h_!gm8u&hK9_5alHY_18|^*)mY-"
    "?cz`dccS}6u+tdnpvUd=tBCPM9^x9gPBth!~>NCBw7H56puB2VAZN!8l31imC{^K2N3;lP$)#m*4jqhZy=rozj|ZGmJ$Bln%0qjJ"
    "3^jJ$~jK@KZLpQc{u)Sg7&ICpu$D^hf+2xpVT5Qz=RuYv1@XSNk4!d;uJIpt5N824U>Gnp_+q#3hN>rDQ#da{9Fa;`zC{xecOhw8"
    "H*V>BZB))BvzQRDYukRY<<a4SwXBs-DIL;q1)sI0pw=wVB@dmJO#I{2&19IkTPB^@N0mn=eSFNi&AYfb1npCbaalB>UbAU#936i7"
    "f?B-yP5rTKf~OgnzC-"
    "f+6?(0zwj8{uB@Mr?gP?_Qa$OZe+L;&H@hyOFK&EAQkQGVc}@XTH%jW7L`JGxJh=ZH?CU#jwfU`gRMQIyTwgcsS`ow)zVF9LNPl)"
    "T~dOCHPJ;NQ*ULukPz7(QEgz{X79ym!c4ZnCm0P%uABU#lSZix7b~}I2F;>2%aORR9ZhvyWv`ckz?$ygks&awf&d;Ul_)KlgWN*D"
    "1QO4_$(wpKxZTGfM)XB`9$9avF{m<X_=20j#ClQw!Cr3VLa4SK6S1%gnPi1b<S@T)F42!T)9#UxHd~VRpCsr76lc*9kWymUDlE30"
    "&MmqEqlD?YgRvXV!6_eN>jS8zYEQ4z{5j}+hs&Z02lAg63%lAf<+QPTi^!Ey>+A>fjo<*2YvLC>LXP@h(>mzZF&6u(CpCp2kS$^r"
    "&1hpb+=VUF1$5YX52m5q9iO*#A6Ndj#%;_r#a3#I=XQrCfxs=?Pa|6ruZ`BVm>FV_@-"
    "Nd$ZK~~DVaIDzQ?nZ0hU<3pXABBc0KKXP7vc6I!rkSBXmSE*+ZA?P5YMR;_aH<eHJbL{4Oeo3t<Zo#UR4{Q+<n}Om%9+Vit?W#uG"
    "5f=Ir2hOtoO1a?wL&`)`ZYn_YtEg+8WYe}~da76Q|DEaN;2GmxxCXJ6BDv<*+s5{=T^$;nF0TLdM{{nJuAgL_yS?|nJF_CmH{0vv"
    "rTT0z27T54NZm3&H0)yD9AK1~U-SM*5QX4eOM>1wV@mR(u^kv&^t6ztBI(Qe3kf>v`CPl$Km-u_NY?(6<3VX#TQJJ>>|t$u4r-"
    "dwiGx_g(}ihcvmmtb;KCMr8a2k|K1sXA)HB|caePuCvag_jUV%^DbB?-"
    "wj4e1_hDZT#5#&x$aTCPL%{*?y19)wwL!D!OVrLI+p}iypzikxykHnvsg};H~k5-yJZUp?SjK&UZ4Nw}cmwhgDj6)-"
    "CcSnc6X<wq8mlEaW~?-oHIn+p%>dADGwe`E|0{zLJclRO-NZU1xfk&^bEp#6Bsog(Rytk^MOmk&p<LQNid_#3BcbHagoQUxG<Qn+"
    "Jg<6&l~q;|?sg0IP?yeG4eGuI@cJIW6>^+o%^$3bEuuz7d<(86BUVhlar7nX!{vG)aI@+iQG%IV&v3-"
    "sV`dw3*U4>DFK3*O8dwHExP)R2SdBldwM=#OK^AE}ITMi6-p-"
    "`_H&s#+fUxSuu{pN838ZcYHKOHC{BBgk$SPr|&l|Y;%jB&+ST@5NDLJ#bC|=rWrJn@!YHTSIH>uLPb<^6%SMLy&w78=Wjp}@Y@z1"
    "cJ*nDxS`X97dDc|=+Q4CTu*s%5%wFfmBWr3x7&piq}hon&bU3QM_TZtyq#-"
    "@lV2&?<+j{S1P=>g$;32q=|!$GUiP4|WQ_?RwlLVp_yfN)e@p<9k|m+tXDBxRw81Orhb^gM9SU(KUM3#hbfO}Yf=;@uQ+{xDBgHj"
    "tJ43-"
    "1<}y4z2^9Of=#IwBq*@<kD!3f8R9Ye5#eSe}Z6kL;R?ntPaHcBZ`Hw064l9=omUWUy=rR=f4D_#P(;AayBd9BhcA)PJUF3>!YqvN"
    "xii1Hy8<}&dZWm!{c1`50;Hl3qF-NpKVE*}RJ!2vM{i+XO=4+M%-"
    "zR5!WPpbES6c+%$G*OC8eYF}JoeQ+@8uRO$v)vI0C`p~PgcX7Ah!9F-"
    "^9a|yBTED)UCLwZCMDZG$a^S;bRotufCmPJ;VNI{J@tK>Mfk18LFI4Ym?(-"
    "^VKBaNRdwlr|vAbwH}#R+gRw%Xln})Co16|E*9$7(WSiyOHKQeKiw?;jitQZhdTI#9Y*j}8KxGe#BxS3{dgtVZA}cuuN(=XnwKfs"
    "`w|80us^<^+bgn_P7eB5UuVij_tne0WQ#<a$#&CoTAEO;lf;z*#;OLT_B_Gdj%oME^dMw5p#sNjX{ead8xm&NiyzHCI>}oAr?Nor"
    "uj*=}*8EMAl`?16=y$Uuo`eZ-"
    ")?CSs?u{ohj0I{INF5+Sou|IF*lpuLJkv&KCbm7hO>&DWnXfqu@q&Sv;y9Ump!X!w1fHI*dbX;m%>5O>;I)!9pkNiHDw#+tdqcl8"
    "0t3QpmBNr$SIA*x^=()&U`3cB@w)orfxg%RRR#e)o1Y}&+n7!TH&LZz-rQu-Xf3l?4K_wh-XQi9{gLqOw>OrIcHVbg@pJ#C@A!54"
    "1vmas=^Zo*Pk`1z+TIaSVJ6<eCjXB{)ssCEWSBUv)!=(P98#w3KZrs^GY=>6+T14IM4MLcFhz1dxNoMx-"
    "fKRyR_dPlhr)Gynw=5hwYhU`KJOm^B4+bDko@#cWkx}sQrv90fBKC*Y;di)jxp-HM7}Fb%nU(s*P1Z*u*1;+EI;5%r3)x?u68{cO"
    "M4v@`L-@2(B86J6}EWS+096F64<Z_oyR!6w#pGOIqRk*xpIknG(~<IUE+la5?=Ab(AS9Zo0AFIMKRis##TEP-Cq;?=o_w#-"
    "Dj&qZ+s(kXMD9BKZQdS=h;Rq6I8V&e8&*VasN99)J%aL8auXY!i-"
    "HU$DPc)7S*q2!Mit1tG!O}1&m@67PrI_%qr*mP&8O*y4%q&IovHMNX#aka*=EggC{{d<%<mj)htrgu)h<j5*L`HS{K}uS<AnL7Hz"
    "fmKAC#Edxfy}m#Oe$(~fXzA5S_LY_PSqVI!{z+d&iWq8ppuwWJ$}2fkwENP^<8w>g-!`?wy|yEc1UlU-"
    "Np7}Ml!{fp^Jp`XaLGw#7YS_NM#fK~xq$IL1`u>5$>I;IU4cMgIz6?dy$Bh3^VW-xVk{)O0YMeA;qv=6XG0sg&!N-"
    "gY^^rP`MLbUD4+4-j-"
    "(rXITBfmvuU=!4MlnfGzi?dgvbU>PC)0y31E8CaP*T}k4lJ%>2V9R>{ZZkN@#5uwYRDJsDzn}&rgH8P&fzz#MKJ7ePmG3ikx>}<3"
    "Hb0<g(@6m?9a-"
    "aCObh38Z<~w%nU5(xtFGx8gfdFosTD^Vcf~Db4*dQrrOTFq8q45fpP;*o4o2H~R<LIB9<iN`pQX(eck)h@SD@uR8OmkAShVMJnC<"
    "-;V8|8m>^iAO8MMGOop&c7(7N~>LzQ|PghgP=fo0^G{M+BSQ(d^E;GO&0u+VNY4gGNr;PMrWkWja7XTJw~&DQ&h4xlYInNtG}Gww"
    "c}LTI+<54<dyYnB`){++p+<xc}?p|#RR`7UNKV3|xSCXsUOXMjh779H2?8J-"
    "P=2MXv#dH3{{h_{2bg6ZdN8C~?2nVpL77`2+FC);Q?s`qV%wYixy!hopdP~uy8_;BaxZ_Br%m3n1bAv}2aIR-VpSBQzd%pt4PStZ"
    "P)2fwJdS0X(zTuMbrf(Tpy_BRF>OO(67;lZ^`Ac_x;=SI`F7KWU(f+{S}aRY?HjXA^SDO>GEJIo@QD9diS{aH@}V_>EC+0}YJBkA"
    "=l%q)%eyx5PI_T;x|lP$yK;+<cATX>_yGY9)NV*pGbbCR8kG25Xn5f(<5XFy{;zeqP4X$>lCbLL{Vgi-"
    "U&4K7b@CKV$&)$MdJ+&O(~v|&e2=I>Zu-#n`-`%cyo{-OI~A?eQ9ac<G;Hr?*YC@R6EOvpb;(vrLjJxF-"
    "a9W`~OROPVaf*I7ik11Hd3{C4L=b|^4k6d^JG?wJql4!G^e%3V>a>d|Zn<Pu`aYQbC7t6Z;^OaBP`-btp_X#{cB|BXS-"
    "`hquZIo_y@W#V`%-AW`fnzI_XfFnI&87(Z&y)B@=Jo_`d=GvNA#JxDxMPQ_&5f{m-"
    "}{a7w09!P6>+pa&9rW8BTsT}#XzHrUc)90>u`V47xcq`j{+Zoir;n37FScO<?pin7*BhEQ>7B_kW_iePg3Mc`FBKi2vndvsFQ>oL"
    "$=l%dVH;GNqTjd#5U*5mq@C_cG*%dYis!updh0?!_zA8l6I6}2>!0#x;uViaCj1Rp6gOazeLL$QFwm^I3{HNPXtokFWDuS8+pq6@"
    "jb2Eb>RfdceTcZ*EU~JV^?qOICqR7g<3*mb~La|0t-"
    "kbYFR)a=442Ki+<lr&!)H*34>kQOGoT5EP1}TwwB7_z4A}EpUFQb+!4jVsvyGsqp!v%+^8j8TRl4P#4@(e1Z_2F+44pSgN-"
    "c(px@5ti4JjuNE6!iwIkkTJLv9jY=nz>7m!9T_@0b}!^B++)aK#1K@GNq#x!3rx)?=zZJB#)EethyG}$*JZ#&ltX{WZmpJ|Uw{{4"
    "TMq19<+>);K|lR7)3<YuMuQ2y1+eFl}c9my|$DsLh{5Ob(EWQcPT7OXo$gok7u$Gdf`-xaH-YcsVBa?>6+QH9YOS4l6kw3oEp+Kq"
    "JEmOF{3ua&OFYA8Z)PkXGApCpd`ZLm3PL+#yS1|9{BW83rdY`;(nSCo4zG_k8aZd)3A#Gl|=`z=PpeE>TYG;3369pWb6_dD@cZmK"
    "JgJcO(!Ln&8Wz~tM<-XLw1wXa3o4adK`y@dZeufp+$KS?$H1Ax-"
    "#TnTV{dn@YGih18StkP*9WWreaYSiypt!xR$`w^|mZFW;?G^H&S+!}&OTfAIU1Roy2vlSX~eiHu$Ro_>$y{I;63-"
    "D)Z58G6}IAn;e&0yj9A|O%H_lKS!>8R7bx8UcXOsolkoYmOVe~<nP=jIala#fB5(j;fH<;^udcK3A%SDV?!drWvB!q1}u+q=pRm^"
    "YGVMc>_Dd1YbT5LC3Bovlc-qm8h@Z8%4w4k$AE(&(q?Ig@3~v^ta3Gv2E!-"
    "kqakRmhw1+IV(<cHt8Q{K?=<=iWOJ0|a$DcuOA!s>F54Byah-c+=f2VKild-"
    "D^>*YwHvR!`Ax`o<N~xJoKp`lc9Enr*W1ByngU4lyD1>tJc;|g~J<i4MVnmoaD~F_m2r|WXQ|Hg9Zxp{Ue$dDyBCE?L4K<*o#<7M"
    "A+8sqa%hH{yrlgX9XLENgs3IT2y7D1%31t1xE!47~1cN3c25im`Yjee>@XLQ9Z^QCrrpvD6A!QV7Gk>+RdrjyZ~qAl@{_<_4bZEL"
    "_Du~MXOIMZHac!zPdH~LNtvIb62b0R$$iNM}<!|QAeT8le^2Rap?+g&5)5~2ts4K$C9!YamzlmqzRr)b2Mk4#Xx)K95W0Fex0d|-"
    "?eC;JQ}m5=A7nBi(9h8$CU~kqmi<wFFPuGJ;@c@c!L!Fz>Ph8X)C609qHgchp8>?(e{u)TPj8GnXLy{Ng<`H7Y;}AYvNCq+hc$Rv"
    "^31Jp735FAab373SF6d7dHOLvR;1OIF|#7NPtUV!&TPN@qIsc3XP{HRtOnqGPk}@iIu3WMxr;O{*ixiwLt5r@IC~(WmvIR$%1t`1"
    "j+<YZUq$hT@Bkdj#|M{d!ivsFU~3qnSac~!-HG~-|n4FPltB-3=xJ67&6^#uml8`o5y#Hbk~D;g+3-"
    "sGHBE%_1Dmd#^Q#8TOxq1T1)-"
    "1txqRQADPlg3gEA6zZ)dKCJ$(6xL3k^Be9owXs})cD4$v7hJtvW+l(ZQNAaCf50vOSy{$E#Qh92KBmsqqv9zy4(;gZ8mAhq|Uf@p"
    "CgZyA|+&k!`nV%#p?>+$lA;Vsxi4&RHkJfieeE*$>=vM&kOh9KM9A$XTKHg%067FUXXq^SG1LzoQ)G@<LK}~BYUUwwrGXgmTVJTb"
    "w=t&eGG)3ILaA>vWmw0+wMbnH;%(L;ynb`NYp^U&uqR5oS??kb0NqS9)m;FJ1?+0zmn^-61>x}ES6p4<HbCi6K&oU|4Udn-"
    "CzI?D@;J-~Yh-Tnj!;bVbl(5}Bp&r!mt##3%VQqmWewW5<i1AuUH>ssthXnx(keGb7f&4xZdfXvCJT0~$y+~VVD!4ZCLD7((_&Z3"
    ";w=5b++OrY+FB;<IJ(u`?Wm?|wB*0yy((gxE=Umj*W;kv8d9?pYk{&tWM)Fx=2@eis6Y<+Ucz!M36&L*ROui7lX38`xuNLUrryc2"
    "JEIfZ%3L)Ab;g~NiOSI`jP*g;9lE?QA*@&$nfsbWO6!8{c*b#S2;~RX-{GC3-"
    "&PCaSLJF`|P($Zm^GoN<(`9tB4vRX>aTdJwrLwK>;rF5`wQXzXVj~^C+Yek`7lN@fxZAcNS}Ln_e(H^Vf;XO^M2M;Gtz$@v-"
    "|mG8OQZI;q1<h;Zg<A&9k(EZn#SiR`&S!CeqKXrsh7(kas@E@_Cz@Hi){7yC(lW3Bq1bL-%9pe31C3*F{k|XTYIQE-"
    "#6*Z{DpQhdztrI2ARb5V2xJXl563uXooJ>1xw}|w@=Sv-"
    "0#kXAtVl{_==V_>FTqYu?~N+K3}mTRlzg(5ToVg|Lx*1jx4uue6~)5wXUifFP!baoO_ckGXDJ-NV;Do=`o;@Lx+S&4ERj@-"
    "Pb8DWo67;i_QF6HnQ+=6>n&YEm_oVhLNT(7)w8Em&DTd%8A)u$r~J&^oh5#Rha|~BEFAPW%@Q@pT!c@&wDT_@K=QiC9$K@BUP@(o"
    "aEiY*~bX2f>#jgUOM_OZ0TjU6cz+uvK@@}J;6#O*wwdr5+Q`>Sk9lkAs^*;S!9m=+>`Ne#v#@k{L%-"
    "pi)tX~aXZWb%}Tu)MBuV+e?DYeneU*I3swa}a$$*;^yKT+{8rv2V<rmfYn{ZHOed)3nE&fZ)ajgI7-"
    "=BeD~`Vpps^6nj)~Q{JF(CE*|Lmu+2M`tO0lO)4l+gq)u?}JhC@xl<7E#0UR@YnLE25g_0g8dCw0EKq_-"
    "7#Cy{x8^v2~a|J1D|s~!U4e8=|?r?-ePA@$wFDj>4sPEu>51Z{aCeJz!%<+~y*-"
    "^fCa%928fS(9(cY<2fLb(VKcI+U1}$|s}laf|vA?s4PBaHig{qn#fILdTfk@=Ufh!-"
    "P_$bS7apRZ_(4`<>j}NmIE4zI;=<ER(an%0CZwQ~TA+KSKj9>bS70oHL1+Yf_!v2v4#u0?LY3cN+l|+^DuRqtjNN2(;wVlI?SMI="
    "Pj!!FGV*12J{D*;k=#JZt-"
    "%pCy8%=Z)5?x=5(jHFyf#)?PBvq9=x3Zqznfqf{)2rD*8A2Ud&fuahzu!fqv*Fr3nR<7qVxf{<<va$Ks@+TpHW#eiETwr}EAfO2f"
    "Xu3&qs0;i|)MsfZQ^&%4c)2r`0S&nX_NGKhZR6c8K_w_#s#O4a_5o_E!q$=5DoP9d%5?9y6=lrF$u)7u@CW!zOgtLclomErmrX5_"
    "-<Ofbym$p&(cYWB2Zxb#YsvID;Y?cE202Xbu(*}k1A%6r84mS(!Z59@0jc<NOke4bqtyxU!!rZMXlX;}_fNIN)Fd2e)y{b4MlD*Y"
    "_-_r!#bVa?@Iyf5YBZKOTruA*aSe?^W-6OHdaLRGG#UWSw&?SklV6t!WXu1o`^Q2ht_SSogWqyVn524sm57Z-"
    "1NN?u*>4xpC?khTykzr~DsIIRV4*EBAH+>Z!ycav;`fCNKLcJ?r3s<r-hIo^ZD2}_~Q8~GWS;Yt7_8o7Ux4=;1Wh-"
    "l3in+sd;HW~!HiZDxC5lJeBch{Ra5W#EF_@IpHw_Q(K|_i~MsSuT)ps2h37-"
    "607s))lX9{*9=2?Dplk7zLURnC1NyU9VptsNaJ|_9y!Qz^KGd<nkUpH=+#YncqE!NpuU%?hwsU>|45o8=?`i79Pj59dlYKeZmIx}"
    "yH)QNWGdB@H!>Iwgddw#hF>E^2zoAVT0zElKpmMyWSUw7vhCW8J>VR`-d${cM^mV5G}n-"
    "dV?u}RfW|MZ<(cVzH%!EeKt@JkFwx0JodtodGMbAX2{`1q1(vE2rgvu$k$eD`F?GHwS*Rh5irJQW;4_FPz*Y~ERC7x^qF>Yf1thx"
    "pZLWZ?)AP0<b3u-LljAOeHT>M-"
    "%MS~D4s%lsWjNF>y+ZmdqvC7rRBT3EnaY=ARy8$0s2&l+@1)exat_URq)d%y9XnhKo5D}NJMrquRqO;>(}_`zmomp2(B7R!(Wawd"
    "jMr`HCiQUIL&--jV)?Wrr6xOUTZ1^X6@_uz1Cxc8u<Jj1Pv%x>O1+ud_Fw%KLpzERv1l|38jfSrqx(;Iqm$wq@TH~p!vS@~O08v?"
    "%*B7@WVN{@JK{krPs{);+wFONvtzpABUD<6dG&L<QhWY%Oq(Krmj8^Q2^^{)h{nImac+edYUsqBu?Qq1}#Q4p3gNBh$AVUgw<jlC"
    "U>;JH62Q>E42Lp`Jm@JI&x*ZE+GqXgY=DB9H#%8N!Uw76b60e8J?kWNd{ay-shh|)JUxqXJGL;x{hf4%3dlfBFCzm$d75D;vaEb)"
    "9qWv;crltg`HIP#1QOLnX9uPaLLK0MiA9B7$DH_a?t+4o<_VBOXmy1rS#+YyQI>d^p*672X5l4$pRu$$c>0%^lj?oRdVn53obnhZ"
    "nFfW;<=>x>h1<7mDo^2<2Y)}9%lL{#!uNcwm0?b)ewmia{SRN?m2^=&iyA$VP}Tm1nqP8v5erNMzH_Rc^AypDqDO1DIfvj(ehw*8"
    "}MGuHYUBn)(b5=p_$YS7lBuVi>lJ)QuohP>suGeyErS$d!>zDy7iwf9(544Jhw>~7y@JBEPVIMNuYa%xjx#9{J%=gh2T;Y_8svhI"
    "&fB#rl)by7?mT`&<DK}5(zA2r46U@_IO-aST0b!4}D#LzufQUD%!s*qJi7G^{ESb6ak{L1r0%3K+K_n6_%8dn407;7l;5?t9EnAm"
    "d<V@QmgO~aa1-"
    "vB%5a?OQJ)p{UuKgm(z)_FiW>%e)laaFPJl>(9cnS%(&qd3YK`VW(?r$D%4m}N}83Dy0`cU|C10=y%wGHy7A6br8y0A09)W8;lNP"
    "6)7*e0Xzc#dlK3NkyLt^spch;0jJ&=U`4{)+d6?9{uGr7K(shG7w{ZACd#L<-"
    "7icg@&xA=;uXfX_|K+{GhKD+mC@n#+ycUu?4K062$(Ugf2*^PEfviIH6lM!qm35NGUr7gKk+pO#vLPj1DwLxSSr$Qe5V=<a;yLYa"
    "C7HKwU9N490`+)Z>HN+D+}MkIpk`&-"
    "HlS{%Q>zrF|=PR7PRi9hBfbbx#V>YX>3Sa`@pd2nEsn{kE{#4+)<oxh<OA>FJb3Oog1Ly+>9(D=of%Fad-"
    "q4eaX4Q_mUUGWh2kk2SgHBg7~)-"
    "Q0QIqneA6gm0!!mC24};4;EjOu^eC;47OhI??arUE7EMgwmYZ(lvqNQllw5eyL7d01`AeUoY`AL71voPLcxnd;TU^I2(THUiMux>"
    "vya}3Xm@pnEr{s{@A0|ei{U0!f06QxxU{HTNjcIITYaTRf8%EkJB27f2%*++yB;<RJVUNY7GrO5q1}S#@fk}?rd;EXLdV6x5UxV!"
    "<4@)BeL~Qe5Jg*4%t`|Lv{p8D4nrxl&ru^h3<s#)nYWpk~rv(JV0$kvtx)ZLxI`1goDh{NDuoL_om3eR5Z5jZaGI|)F8=gsFZp2+"
    "e2j^27ODo49U2E-"
    "hh2ee6&Xs#5A_luZ?U}o7^n)db9Ms2pGy57&%aV$!NQ6NDLwyCN9hE<TQ3V+5T1pWAJjfTjd(E%@gye!N(QUN)y*pIERN`N?%qx5"
    "{s;PO``f)c{2;SGWFT-"
    ">fOK|lbHi#)vzUZIGwyvm@wOV86qr5j&q2##RgCkD;BjEb^5p~>2bJTe@3B1<ESEh?sNa!i39GzAZ^V6@uXywxo1K?OLT&GaRc^*"
    "PahWk)Z|oh)O2)<HE&e^hBnJD?kA4s)B0lK4@lmwkjrQ=23KGWBZup>s{g?edD_yyNe+BFvz+=%Wk@YrjibeY)p#~j?G-"
    "*KRM#XtL-mz^m?h^iVcjB8{pWvd|DonKlQs|dx~1{%eQzkCt^ziu&!5WUh&JdugZHZ~b)+C`qW(-81~XL!t%Pbz@oDzy-"
    "z1`>u<m<!e{;w^Rn1K>yZM5KVN&&2hf_4MAFjjG#6QXy-6G9weRx!!>xvd`*Op#I<;w+txZC|jc%kM|+|y-"
    "C=W$?P;2|k+6?;#Z1;^S6a;x->Bzs+z6y7bXExyw5_;IxnZ4gq-<N0)|dQPl`r`<NYuHO5Kj5;EV-UYy&<xSp`NZ6QU_6+Zh3B%Y"
    "vb{GIVXFbrp{mz=iO5Ntp?JJ|13fFZf!Me`|{F;=Xq5_OlCS7@tUCW@_Y^EcmF$&%o>_WZD$89-"
    "I_+!NLlqxl$GfCN9e)XOcmf<hlzU}xM$MRyjxR*cj>=MkVVfXt^<lNa<xM*-oQ$`7zZ0_BhYZSgu-9chrwmxIC-oJy5LMXmw*_|A"
    "0kLmR@Dw~O16j=J%-ue<6Mm!e6T4xu<g8|-fZEX&VsL!`e+ObX8i#sjLD5;E4hjk3Rm4cE~n{B#(!6L~31Y<1c5J4D}3K-"
    "7!2CqF;Pn_eMtdMcv2$m>I9lpu+%h{8W1is@3o7fy#2xmxl&u_X6`OoyVl2~>8o>mC1eGlwurYUET49W%;(Xxyy1h1?z=^`7J5jO"
    ")Y6}Rd>Ik^nuD3qXaPtQ`q8KxLA@ai!jDd!e~lmYGl3q(a3-Yf7s1N6%HKDv|s%xT7-"
    "Og9J(&dKc+xkB6_9B8(;dx>hC%Sza~lI_k__4`$6Q)}uKv$IpZS4xpJD$`fh^xdl+(5+m35){8w>6ZikO!x{<{PQ!hB!~6TY=go~"
    "*p$y<4?p7BwjaOSzjTJ@pIDIzaLhVF4P;Al4~|p1gbn;!GA03xF&Pz0tu(!cjPB*E2b%)=227a9WY)dAX+v+qtCk`MU6b@k{;#RX"
    "{We!YEJfFncC=UK+467xV_)XXg^0KBQpnptx3M<xU7M<d)vrG#bvkOGP|6a$o0}X7rtp<gHuN+FJ3E&+Yv4VJcjUj$H*q=r=D!KI"
    "qla=jHHk1U_Z??JgSKtoFzxj02}kG+&R*IiM{d0)TXc&@%-"
    "F}Esx4Cyt<R_ho>8yc{ZZl=hm1t0&?VWD#Fuk4i0)PxS{}K#sz853>~dq7bLKnuUmk?R=I{G+{1QGQce2aQl=o&nGb!CNr1nKzhV"
    "2(|qQOrWC<&)wNc{=+In!^_4$@fTf(nhxjGF@e{dEd!B8$JnZB3Np*w&?&p>(qgv=f$Ll-tGHj;^n}K>OpQI%>7*U5>!sdQB4Rbk"
    "r)I+9H!%*A1MFezr*wwvX7v!ow|dhZI|{nQX@!I43DY^)HHC3bRSQfBrCrr?veP!^2F~w~z_77WUnp{6a$F-"
    "67SgfIAu&bJ8VnsjC}UBrmBXk!F_F>dCE)C+M&`XJPT!U7vwr)Wk2ApI4#>_0z-"
    "*EeU8@J=4QEcnD<lm~TZCR~hcFP;)QkJP#K<_)JT;_1Df)?{SI-=k;_O#s-}@4kF6EL&wO?lP-qfJfo+)rBDSi=+^VOXa8@zO7bz"
    "nuGQ;9e726D;?9xwec?>@X-"
    "PS|l=)<=_b9)=&(N)Hj%r`s+{R7m6O~AO+x5Zb$h<<4C}6^pkeV|O)7tsz<kbAG8VzIZ*s;_BhZ~~;RO(Uuu3S9h|3h)^z=HEjT|"
    "9zwFCy1%)I+sUUXqNO-pJ&p%_k9~8t3mcmA#Mtx-2}kryOpTGOOSqS@r&{o%_bwe*I-"
    "D_m_1fT<uE!_50NIj0cc7O$UGP%#AW5x8b*af7dSMi<J|W+>g1nkj|`Oo?G-"
    "+6?4!d9YF;{Ah*CdHLp4=lFe47z0L(S)PBwgYDqSa$;XPbRvq>gCT}v7uVyx~^<(N0ESysvWQmkf*UqEURLx97%sIC->CX&p)-"
    "cMk_?u8pPC`;~?qQsF)0Av?3kC+5r+K}emm|I)xhPEu7uzOPEa?c{Vyt(@AAY^qpXYbPdsJCd3AH@kUe$;5Bqn_H9dF1S2o%dY5t"
    "nPQ)f|`aV2{6Jn4^_KYiX?-"
    "&(@RFBq;b&mO+WU!AH2hCR^Q1=`Jd)qIOcd)x~d)y%C{UKkQEwInKob$QylcVbeJP*U7|XzsuPFx12JkqK1dXImV9~acN9Y`!?1J"
    "7m<4|&hd4$P0l?4iaS>4wr1d4?chWEL`t#|D7EO-"
    "e^RZ8lq;3>wyK$4?~2OYn{e(Oj|vx}3P8$l8(Jh`OUNRCjGbW<5YH3y%5xpP!%u#2fReRPHl}nfFG+6bEvm1D;xAv>`mfAP=KW;e"
    "jpRQ$ya#cc47=Xg^EPECFNyBzM1x$Kw&2k~{+=`kFRgH*9H-"
    "C^RVV|m<({tj4880~wSm@A*_EvPn<v}IHd*C5U<E2E+dV`;HdyE38~0Wf^+r#RzckW~Cks-"
    "`_ig_vIck~6WE@O#fZtv9kw;%9Lu*sq6C#r6ZSLy3GuEF-Tk~hy_dQLGC3gJ!6$`|*A@#(1HZt}$_3xQWVD2eEsG-Vz6b1FGL9i_"
    "Cl}5zSvAg1=b&{%Om_CUtx@BmUw#z|NeOMVh<u;nwnRE2Vy=8cAH4aTPR<}K5@hD5F?7OTrRPfTKBA8y7Wc7;7YJ2<S7F~NqKZMk"
    "G)ai%Y3-#bB56LUfkNf#HyN^t}+>U>nyLs-pFb+uS{AG1#&}Tp9ePO|Nc|c*fCFcD-+n1{GgD;?!^A&Y_a>vlSXfb9-"
    "7=T^kBCS4pqZarFTgd%pL^${<iC7smYz)ag?=b~`{8Bh6Zx>!qm?ja!Uq7Za>pGn;_-"
    "ywVA%=TOk<Gt@rCi%vFj4;XV2XP1$Nu^nI9l#qO)*$$-B}ioG8N_dveXNWXsg*!Ac+<wey<;v-"
    "gxBA&1pqBov`8#UXJg(qc0SNcMN`QQ>u|2JDw5NTaV<P08@z{0cZ8DUXgYjP`gUK8rXO~+0oSaSF_3tw3s_}>raPN_}747D8G8WN"
    "9?x{8PfK=eOE008uS~ZvWA@R2^c=|B0uKf{Q#Ib*pTD9{$yK9C>rjshPq~S8~W)JTtQnl4DVN8K==o}y;+0!bNQTM`My<e7=TL_2"
    ";L&1RSMkbQC|&S(+VYH@CR+7H1%3u8V$oy#KjFLJusT>V5l%##cBb)6|lcVl@i_wxh>|JhT~1Pi~)6pU)mdwkS?FvO9OV5F>H<Ra"
    "7ocQd<5_PV^Mjvs!^AzejqwOm)@w5FHj*zJhqSXHi_`GR%(}Z?cN8x$2RDAW{wf`Ao&8u{jc-"
    "h!kc;%s@2)msnruiwxnk4>|Qi06Tn;R)F_vA4amXB<nGP8_byI2yBq|<>OlN-"
    "o=ywhiGqzvoknR7OKkAy(SzBoGMe<5({MZ<WFG3C$2)+(7t`vNZK!)WTi7E;ign!J6#LXfJTxI3WM2UVzQ=!YT{Y~jNP={{lZr-"
    "x_-3EDB?GBb#c|Ti0}(x*UTtcb-zcDuPGT~vzZKO{7+lLZkS%<8LPGCbCrvQ!C<2#cs82FHcD;&-GpRx-"
    "myAxrP(bmz#$9$kd9tE)FQZ&xKFGSv?Z$YOm1z>4wwWRyX(O*hYhJDFdZN-Kn|cF8?tQZ1)}F(4Da|MD(h(Y{v(O^6+m17Q1J}qh"
    "lT%Y*E5<j6RYSejU`otG&!3{?2u~cWbg86_BL5IypLuT>KYQpak0=W`_Trv|?ZIK;3_sUz1b8O-g@SUj&uE#D(biB<e-"
    "x)D=4l5cQ&rvXBrsaD)UcginaGYc=R)3!$>>yzN7K?9+%;be2j#kzHlf>X)Gf;}ID)!WD6PpVtsl#~$Ap&vyf;JULiHjkTa@iFfZ"
    "p$0o3()7jLH>H^M)ULyiH`U>%R)YJ>-ha4=2M&8ENaYyOo~1C#-"
    "w+7oGG`F;;8jeD9Nx_=#BxaMqsgEDUOf#om&HJ=dy44QP+FBzU)dv0f&20DXvwTk#(ZSTBl+$$Im2kxTj9cqpqubjH40HXie(=CU"
    "-8V?1lP>|IyD$%N7V(99)aeC)asFkD}13UncHEE<*QqdR)_in;XP8px^4l%tz5f0w!LRun|DNs6?NKeDMZ;Pxsv>fwrKK)#z>s9q"
    "$JYgORkyqrEM!51J*FHz<@JSdAp6J?Xj(&i<a&SkwS5jt60*DcBGE|AhlOE1bpa%ar>GXCN?F%oGw&-"
    "UrQ0a9>#*m`$LC7gNU`gpCY(lmu|z9kH9eApzkrVxJUD$mUjt-#-"
    ")IY({BYePdps+xazvW4DOIGJaJfM^25W9Vt+UmpkZS^u0T{by_nds{K0ih0I4Qc~k9cEMxt5D81gZ|R>gbw2mr8BS{@c9G`lJD<!"
    "PddXm|3}Wl49h&E1f@EZx;YN)~c3<=cMk4f}bI(K@=44}+n;>uBhmJP$<e7_a0k7@(GuEalBp&f}<MdrdzOeBj^EMoP-aVVDHvBq"
    "U^?wCg;p91n_r{Gq&LqF$wGA16OJ1HO;@ay4U%Yq`I)2~OPiYfAu5hJg?;$y(`a%b8f8A3G#V)(n8;L-"
    "le<XiP!s^c+Tu_BbQc8b0Dry!Cm)4QM@fK!<k@jHtx!T{JALuS{0Wp5>7JFMViUv8UU3A%{?HA;v&!DD0hsMYy4f(<Ue3#&5$*Z~"
    "q?y*IA(y$4?XJYp_{-(K?S^d~^+visEz$AKS<3+gCnVU@Ho288KhNjLq-=x`+KFy%at$*@Uh%V?*SS|OSNgTN7`iq>G-"
    "xB4upRMJ5k6{dfw&S$w{SdMY5YKE(XQ8xbJEjo3?w>iu=KD?h-g1nB>xsshk>Dr5+(IJQI#8sF-"
    "&4GT0pGDrgmayqOGQpKb&(BCqO?^7qEHVVxsK?3&crixAeSjL#wnI~FYVR0<ay?@*R1Hbm_6>m@qYR^)#&-"
    "uOzfQuGaVkrhhbExW@BtI9QR0nzi<~=LYn;O(XWju*n}=189S4jfs#sZ@sRMtUQG$!qX)PETn1Wuh7Wea@_`?o4FEcmS&I@eRCYs"
    "h5Ih(}eZsU@&e{l)?W^0B7_}rdsrkv<8%PzG0ZL4NKO!3-_*z!<?;~3OLzW*MpB~|Rdww)$D#<X<6R5lWr14-"
    "EE?u;SVRTW+(=D)8{1R2+XTIkO5$>K@M4+`OmVCM?1}1ZNPrQ|#gK2DyC<IRh*|+>On6`j>f-"
    "7~iqn<jV>87GYGoGytcmQ=i)cVmFmjtMjU_n&ARUU5vR>fJmyArm}cvw^Ehdf-YkhgPYJd8($z36|}7!PbB63TH`cVQ2;)d}aq@~"
    "G&^%$7RFY8P4gCfdDCQ~0e|gL_1+l{3n6YvAp2tW_Ehz8t-_(M)4g_%m|sa~;6B-"
    "8ETHsDNdJqM|*(IA11*nqWwS2o$neVKs<Ey&$e*4eGr%-)}JZ8$4HI%6P{)iNeL2+!%v^5Ho0pdMBYK`JQ@DGcrQc9&<Aqtp2`Y-"
    "uzYKUcQlxD+6Q+x)OuE7LvgV8i{DFI&;n+j}W;iJ4Wu8)WR!DQCH9SpfrbeW)Kd)tg4M)n0>%EaaWj<{^7D%UXmAB>mu?hBW9uV&"
    "+-1!#@{Ccz@SUciMzKCkqG{*Ra$OgqmuX?0<;@^>C#YPM<zFuBp5#}CLwt{rk7A1G4+hBYPKE2<gyl-V5tv7L;*>R<<-"
    "_8w`7+wN9`=q;~-CVu}8p|rwxe-Yd6QhYkJPn;q9nIk^XdMosGKARfk$221!nwr)M>8Pvx<Ry%4nXh8Z{d8QJJ8BYvn!7`~#+Go<"
    "P4HHUOUwT;+L!`oTFBva3`Xfrj5naa{D;^*pJ+{>MGPBTy$>sQf^xMq@qTb?{$8{6)ksPVBtF;47-7oJd?6se*y-"
    "t<H@!}>mD=eE=Ld)BHuvs9sekxH-"
    "*$D4ueNRp5GvXWIp5A~?AF+XkZrQAz)Am>NV>2bPPvyFc&c*e<Q)&<EKpV9I&Pp`T_TC20OK8-"
    "b*rd`tSh=Di`B52N(^7tj<pa9s&ne{e?>$|FUSiJnpU5MQ7$n2_)KiP$xES4^5Fq+X`3pw~HYfXkKp+m*68&Nh;>Kz+=8j!F_Ccc"
    "T1DBr~y#&Hjd_sp5IZAbu1%^eKI@kI6nhP7>Ok4M}Av^+<P@$dvRP{;lk&zKNszkC*iS1DVMGkbe%isPz27F$*0sQx$`E$e$cGBy"
    "n*;+>=mi3M8{R@ahp@NMCGNo`I9ah(Lfh;TF~tJ_;L*7$4p>Ql|=dTId?SODIPmg}@izRZ~=P{FFm>bcv$4{Up@7gYinYp)(8Avs"
    "yT>#`n@v;A-RPF?x1aI1~B2C8aBwkHRkv#g%g;|zgar+&5g(bkLGYfo!j5q71hHnjH_l|YDGL8h!z5yNu^7Dr-"
    "b4m_D_EW<P?rikCe;sfpT6z_OVwbP!$TP}(Eod9@kx1b7~g>Mb}w={33gX#&QckK~eF5U0YrN<8XwYEy~{9+qkv!m<h%xEMQFf*v"
    "p!|svq9d?9sWW0fd5=VQ-O5A=2z*f&L(^wlTY`_n7-Zl86C9yJ>1B=|F<3X2DEJJF7@p<n-2;E2Qafw)bJGSr&k;X{Vgr{U~Z(I-"
    "gK-Z;ZeD>176gDY$tF5+*sbx%wfw56knngv7+p)>Aq`vY@O^a-"
    "odIgawe6pi!^8v#cQtTcxejhl9MV{QWH7zb?*MQ(Bv)l@WP_c1wFJ^_d^%tc5W66?lywreqNgfYTqAsn`MX;)?OIAx$fuTe-"
    "9H|6u%#?*Q&A>ip1FZjTdWADYa^8G$+W|Q0P`tCm3(q#W!<S7FHsnxl3|8SN!f{#J(n!6a*ENurRNcClBk{G$p3bSHqP(+TLDg#f"
    "BNz*)mA~bvM{rA+M6yPnyA#<NPnJ;n_e#dDTyxudr;lxYxkaM^5WZy4+UVw-RQ7gWRR*My^$TI}O2(Ttta{Cn*XE-"
    "680c<MV%L;NZAo}=$d*zh6V26Yb|(YY&hf}oRJWtj!J?E*ewlB_caw!Gn{VwGcgi5%R^>uH`}{uN$?vWvUmJ?Hooyl~#n6>>5{ly"
    "gQd@8nR+hcY<2FO=M#T{HX_)H_i>%ouF&Eh{*2YPYH0u2qt<8YGuGybvc;2RJWZx{6Ire(?9*?yB_#G`x$wZLmlj<BZ^)YF#i6c%"
    "|-D2&>q!|b1$xOZ%R8JCL$`C#4r0=5UyrPq(1{!si50*+Q%e4tlO(fR8k>C+0WQ4#-KdWQ!14$W)=RTWS7!9X`Fp{-"
    "q{H35V?vZ49SF>P^ctblsVLpsO*yjx+X@xxXBC~a_5*n25z1>8hWc1#9XEyCgETp%%i~HV|k?-"
    "SpnWvC0i^H%cql)e!@v@~+NEiz<=y7-MngQ>YX8z1dxi%m)7F&M1>w0E7K)Sv}VK&84LZRdGPLS<@=T0-"
    "mHAtjuGWOzjGhxzwRwY8Wnr>8s)iD5xXSnf>b7%MZ&ta0D$kAOrl1aG&A<=nl(uVG!J*OQwJ0mS&9@Lb{q+VK)8t;V;@{Yl`wdnK"
    "(oP#X$Ps@uzgWkjl?I9=TU*z7n_q21{2u58N)#XI8-+twjjUC}7XHK;Nz5(Xh^ZW(XZCl>dI6(&-"
    "x<|e5v7_Q_aBgD7!bwauJlIme0b~#>=*cn<jpyaO8rCFtfxRrl7^TaQAroqrfySt0_rSP~u#T-+S{-Udg0;WhHKSA-"
    "5cxdZ!nN8~4AQEykeuzuwvLab{T|~!T_Wcy+w|J<FHbLh#~M4#JuXZ87R%!6SH$QP&v-%ZK#kAA6PygXyi~1-"
    "T$8!O&28B>6}W5Aa54blKBbnZM7JWv!eUMi8)aN6%DXTELLu2D?ggl6%F?BDp(tv8*Gi<HJKzkrsm^9IA3n}4^TFOqo$=itzUZy7"
    "cG{nyVfd>XeZIz$Zckk4dZp^i*!{#ep}*)Uq-"
    "{uHm+5=2!E+vIU_oEDG74Lpoq_^LfZj$n$7}`1^QgJkK{AL^0+AtJT}JZ>5q9ztrt$9{L6+hc<g+lUns)A8PFlj%9x1!qP*$%wF$"
    "wVvO-#a{qQGhW_LA2}*V6%;MBK(yHjsM<pWNAhyV3TNhy*|gc=E*PU&8S3Q&@vpX-QY7tB7y;K7;9H>v=+Eh<(Aw{L}lA>CrOCLm"
    "e`e0^et2XL$FG5}i1a$@PZ5TI*N<a#e~cP>&fgpq7N184~xH@(Ob4rpbNwzRrl5?+|o!rAxP3T3gD7n4Wsk1EcHai!fv04e~M8l9"
    "XBWp0lgrCyC(L{<K;Lw%;S^2I|#TfeQi9rM-aFU5-"
    "ijL7Wn0{nSZnNszz|ar5mG_mvn?!O`rUthepOw2k6JVb*r%%0#X7q$|DA9@+o}Af<IT%-"
    "`7^G?{kf9na_;iMzW&EzoLOP0f~taPkz-"
    "Q67BPyrH!sZNEmgfTglUL%p^BM~SXhVlcX=epd8}Pd*CVJYImLADUw9_crK<=PDpS>B0QxY((--J-"
    "_GCshGjNETEAPY%>FXJ)N8@;cJu@m`Wh_o{&`gS~H=R24>LY9I7mspb=1?Nl2R%l6Qb2?}|aSt27cs^mrA}HY`klCnIf=p{*ypBz"
    "uQ4`d;@LwKDJ0`z8Ol_&f%F=oaB_A6*t-O4a1q`%JrKdUM{L-1PfUwUC(Cpp7ge8>hqfzbC~fZ%XC0Z^Zz8u75F^--"
    "K7~Nm1r2sryR4Ja9L0|DbLHSJV!`R#{<HxO@KmH%_*DlygqGxH1TwuMz&z2D*Exa{7r8q3Td`3-"
    "S!rY}By<^|Iz!C=tJv<~upK?cr8bq{jrZ-~XBU-N{t@JTdA2)43p9wY?<W?-W`zI~2<s?O^Z2PLz+GNA+o-"
    "p*TaaL(njtzlqcme${Y$T9*jQyeAF$>7xLjLaXP_K#;FbR{$7ifC1t=Kcnr#mX{m|sE3|K&Lg?EfqjQq^yHSMiUHcGhv-"
    "@>BGz$uJ#ARKgf-hAc;-"
    "szkbP>Brro*g%+$W0=1nRD3Qce0GFy+B+!@+xKvo9TElWVFze!t3?7m0+`cZLYX=rAQP;42`$xe0+$F*$=)L2gfTJ3@pX`3q!J*1"
    "!)KQtRVl#ve-a(8UR=|O;C3{x!?$z@9@@Dhi~(K|W=LW;`ko#EY*_%RfVGmxT2f9^H-"
    "NPl$pL1ZYGSOWFyI~M2N6p4CN)!m7T0HJ7Ra1-"
    "C%&>yDT=Q1F6=FWx<_sojcu4@3RiGl`weTrU}5_gRm1B!&!yX9z1Ix0(|(kAgSgBl~)bo9cEX!T6s3~lZ{oH@NLbv)X~_!Dhi0ce"
    "9E|78uo=3gG}yIVjq{e-<fh97+&S=0(dv?AMk^^qiWF2pPyj{WZ{)Q+&v->ox<qx>soL-_8S-4t|5?vA-"
    "u?bRuxHC(R<hK3i0??ay|4CdZ@jU$T%D{~1B`WlcV9fa3DTAF!9O(!XG1!3P)18%V*M<><j4muWL4OYz8OoC(pWqk!%(%VdBhaiM"
    "VW{V1E@+gwewtj!R=@wdD2?4VCen~q?%DfT^xXsMnv1Zjsdo)@I>tz~T$m$LacYK5xjK~N;-B*@f>6%3<A7{N^m_J{|qwPP})<-"
    "I@Nvo4xvrXE-@`P@hhhuB}stL|wppiLpoeil&R0U-EdudrNrl8Pt5lR}P10tyTkn>bin=8&ai`u2eKxH|!{WQ8a%o!^*VZA!crO-"
    "hMEYzntpAq!?v?c|_?My_oI(ghHU``Ta_M<SiEI^4!$d0@7tF5_nDV|}6mMqhL%V#L^^N$6(8b&+!lL1!Y&fq(K;1w$AXrz*t+^|"
    ";!qU~>&J=qMVCPXJ3zmc!-GP^eBY!X_oU^V$F;r1TNv-"
    "4^T>SsDCYY(f<yBSlZ?lE6Fqi8<oS+meM!UoXPwdR+$jFiA+?I%+a9u@bDKV|Q$NYCUZN@Y31<jfq5$#X>aLgPx{k^X{iyI)5DfD"
    "U>~*l_Qm5_R0?X^Y$_dnD2l`CQ*mRrAP%MJEnCF(&84>#iAko=cb(y}>iKdYdq{w<0U?8Zt|s3ZL$_JVDI<YZJacVE8J$&!G{P<x"
    "+GZN=R17rL?CjlF;cw+K(Q(zd<T5XC~hrt{dE`T;Tit?J|)n2s`9EF-=r0RN=mq%j-aiZ2!=7-lKv~Cwz!6am%MP00-"
    "0ZO5sDATpNAM_)k~-"
    "ri_{JSoeO!O}pl&<OBK=zq#^vZ)jWc8`oTfKU95k{fxyFw<YJx#{JF&$8}bpg(J*9`v{P<)~N7@v4jYoFPG1fh}38Y;FQKF|7O=<"
    "SwgpYT_b;#tAr}?y%(MKk)*ma8|<sGnzYjqSZipP7Au>z886-ER7YC(B@~z~Yqi>SfpNUbHcF*;0N-"
    "0I&H>){3_rDmoz3@@*?G*;hyx*IK6PyWnTq=;-"
    "h2yB#3Ww<xV!J(^J%%t5>HyF8a&+13HkmXN?9pzpsyX!a1{M0k&=bhmHraAXyAX{y-"
    "=Dydp&?XBz}@A0LKSsboW<~5Uz*Uco{t7_blJ(B!zC)rMQD1!w*<eZ$0wPXh^$e-"
    "nw0r({x1%yPFhM*A+$m8B4xWqGjm`7CJM^y@p}Bnq<xHYV9?0a#Rh7W()1QAXOEZ@cwXcx^S5x*2BLvGs0?@DJi3zy&=(i#m}%fE"
    "88lB&R}TiGoK&5@=>u#-?@GaoUiT*W%4as`tzG6(akENoaUoavHH9+F24*A{%Jot1(GU^TBxv%emu3DuWSsiSp*!h<(M2iKTP@-"
    "lOE_q^`+^fmUQoi)PstS`zFePp6et_###0T-"
    "!=&U*ct$0G@H+GOdabqE?Ot7#ombJ87H+5w6tno{2WZCmYQ_j3Z4u_1fxp*l53^7$~Sg~rsE6b*q1IJxyHm^;=DwUm!bPgvfU#Sl"
    "nChSrd(pxGWa~2VD6QSi=J4or^Kh>Da*r?Tlj6c#$}?xgS-wfC$Sd%^%Qmc7{PIM-"
    "{e`_Uz9X)NfojtI+511^{IR80x=Wq*O)Dz+_X!;aX&A%m*<V#mK^jPVTwJ3N1a;W+9u;s_w&61@;W;2#A3#&dqy$QS;vST?uq1tZ"
    "ms2^y1&DSh<&kUWOvOjA%xP4)|D7BKwzk(Z)MKho}{m<_mAm?7=o$KV$W=!#~B!$b^3aZXhH8CPLy=$`iqkz5VZ=geXA(8nC)J>J"
    "MNXXz)O(yG7#qf_&DEF5Fcp5iH2)w+n&7S@4s)h1P+m!Lx%A&Wc-Lhd<^tX*C?eJwxuArhOYCcH1rJ+F>Xu3u;%iP=P^u=0SZKD#"
    "pF?;m!$#Nl22^iV;eQ+>KQ+lAMNP1Mh02N`xqxNkJQS@z=8EJVYyxHQBA}bf;-"
    "k_Aqfv+y}jzyAtezmUskwT%r$k4sv|`p&P4y^uCJFQKB)C$I^Z>I^GqkrkB`nDIJ2Kt^R$xud$F@Xev|OL^>WHzpQ+|GfT)B4nh~"
    "toX7uf5e5xsVzngC#S9N4&s8h0Co|<+t;_=_2^lTUuZBL_>4;(ej^s&JkvU~px1AEj;QE-"
    "@_foOK2OLwqkcHuX?X8Chv{E5VWY<K4hC=w}qze=iJpJ1D)UCua(dYY!o4&PyEPgDsKZChlVrIq=<-"
    "t#}S7*~)5_T0<$)93Qscs+K{(lw*e?^HAa^X~7>4a(Lt+w8<DK4V9q2U20>^QtAe9U5^#@C~yAz9U#pTuV8<fl9zMe8@R;Td3$Sr"
    "to{Vh7;=uzxzlBDf{yLO-F)*3($A{fig%|9#V<3dWH<<gqVmDq^j#D(-"
    "3lvUWT!^2`f)RLlS*eyWS(Qh^mwKY*74~DaslT)VmKJ(nx93ywd;=O=?92aAetOolU#EEq2^eZ_M0xx0CvW=SgVFGPR1x5b|6rx("
    "T*;oe!oc=hfK~D%3Mfr6mChjUd{h7`MS9>u&(}mNDy-fz?x&lBY=?r3ACJk5<Hh1iXQK9R2Z}v?7V6(8c#Pez<fl5#q@-"
    "xLSka#*P`&hJHqNI~CQPJe`SNr#%ehCj$4ry!NOH?cos}-"
    "f5GZTKp{E$5$~gBCq$z=q<MH^S}T5pa1%Q|LgzxpQ%s7Tz{ff`|*VIC$6~TU)8cXsD!|rnm<U8%ONautn7*1S;YYvlbEFzX*LR)b"
    "PP$K>GVrF4S1W)n7zr98Kfk)BqQabG7B?<ivvn4c~nSAjFLydVzqNDmtCHtn)6yN$h6d0QA{5#@lGBHbVG-"
    "^hqcaWXB=~G1%wH4`w$5kCu#J0<{5i*^2fwe#aI{@<(c}W<|Dqb;1BNh-"
    "}ynl4{(c1*_5SMjSsQNLF;tX0C5d8JjLhT2_5W(r5yW94GylGyF?LwHaqJ|(weQG3udB-uwoyYfW;$F*fB0<$@A1&W_Gnn!*j&9@"
    "wG1X8hgTgf(aTwer9Szl<eUWzggtYC6eE#)>I|yr?Uhni)DVQ2OCtO8+-"
    "D5C*x&Jyo(0Y3p;Ux%KV^q&J<a#O3=sztoDrioKC_LaYsShlKfJ+RHyfV|3d^<J&bv;kwxqcuirf+DvpHDiIDP)!1Cqu!K-"
    "ZUc)T&j@(xOQCs|zCeyg_*Ai3V|y`ixxv`De~QooLXY3OCnPAXgj=srzBqADxB`WYThm}&1wWE01%5IdISyY=gKUA3`R#<j;(x)s"
    "O4v6b4dcYBuR3I(IZ9yHaej>7Q?$K;=y9`RhtbmEh)5r5{ZE(nRPM=7u}^ab}Q-"
    "ADGM>MsflE1QW;pes>5YAE+3Dg?W~z3Ap1>8BT(-+P;&ax$G-qmjfvrM#js95py|UPR3-A%A?1NG&#-"
    "@Oi9q#J>4oEVardw2WPX>U@e<7W`L#V7~A0`|tn#&;R<b|L;2`|7RsCZHDWdE#J%9xs9ZUglFIJN~cgVDrcWfTKt4%Z8-"
    "w6J=GOn%QAxQ$xu7I0{QWMdCqRO=&g@0Y%87?ufb6qz*;uulhn=~y)7s0HUv&5d*>52H?X>dgRCQbPhK7mv1+Xc8Nq#8wRsAj9uL"
    "QG9i#D?_^k4TRBapn<=5ojwLZ6NFn35c;ZyGyKAS)5cZ73HfnqobRKxYw<a{-"
    "Ha9ei+4Mi;WY8NrneJyvt5%wM`$vt602(8JOGvj~jkCZNzAmL*he2V4U_vjmN(Rws>ohyruD7N|@6P!F9PIKC*qsg`$xOXf<o%w0"
    "us@pSc)@XIu4Y=6tR$&a<rl&b9Q|u{goZ8jSnsXpAd{66=X<x*Q>w?SAnLs!0!z;6y4%duk5tNG}v4WA$#P*otTs>Yq{>9S`c&ub"
    "s%0pBPYs^CG+Bpj5u&6D{4RA8u$4-"
    "4wYYI={5v+v6;Hbuq{+NFuI>0qvr#+Glokdo&cc=Av1ei$Hwr}tLeVYq!S*g}@q9#t(g>kmv!X{pvOcGbSA&brweK=mM+~(NNDmy"
    "~-Ix~)B8)^^3_g$wj+T%qSAVdx9vedm3g0j?&btn4|{liGV4<<r)uokMYtAU2!AO41QV1l#Mn0n8j7?C+q`Z=M?cW~W=FK=-"
    "*l3Dd6t^PJ0tLBh;d5^Znit(z`O1MR;^*FNnnWj<}^#iOhQ69;Nj|uDtU9=*$ytt9fw<=n_vFDn+0_&0f1IbGXv<q<7mFylgN+t)"
    "W{GAFVH|u5H*pjZ(i;>kSO~6A_7co?%Uo<Xfg3rt5x29GKqo!&b4w{>>lcBJnya|Ka@6feH6!tbqU>q3^lW%v5(v$KpJG-B;R-"
    "YB$hD=HZ@P|=i!lBrKirtq%)Vx~Y;m3B=0&}~s7Xok@W&wLCrMjq==V-"
    "+j5eY5Qj&{AbpG3_P{N_YQFfyF+FnMEI&zdE;%JI$NKD~VMu4Q64V_NLsMK#pZUQNa%EIjs**Ew9Y?$Tv-"
    "?nVWyX`_DYe9!dj#|Vfn1!O@Tj=^0Lw4G7Hm}A#WS50?4t>Z1lfhiAswI)?IP{kgx6NJeM(OfT-Ztv!Gc)AZPS;O}UI+6V$T{LH&"
    "d4@=WH8#9V<ftVkx-?I%4Ni2W$r-QHF<@2pZX8?-q?)hAp2R$)x~ZUGhH#mnteV%}lP^B&BI)iOrgict9!a-*-"
    ";ROH=1yRY^SdV{DdFWd2&^f)b!H+?X+P;&fe>Ya)>_e0oo#wy3PNHquB|gUI|lnWkI(fZX@Bnv0?I4zc5MT;T1m0~hFtT{@gHH|f"
    "+KG`->=%p@S>m<;5`i6wo~Q8)%e>J-?kvv4|Ak4!d>)OSDSi<LakRzs@_08aS+s&TJMs{#u#nkORCtZ8vJD1PMDJEGHk?5P|ku-_"
    "DsS0?+%E*%TOJ-W=xdPY-"
    "*K~yxpb(qP5$Y>RhhX@OSn1B=n6}ff2#DsiqWfGA9kQT><rCCgDCv5EIw>Rm=jmI03pZwKhIN7CoHSJ%%@`Nj_a<EqAX({W#{5x}"
    "m>RRU6f(J{=cW2bX^z{8~rpIp|!8Ic_+u@jjnU3TQ{gCRj5c1)IvQWJ%QY)6-}h+fzd-"
    "6?PPK5>(a%%X1i%{@i<KPrs(K>Mc$18T!xt*9>f__>fZHsNQ-"
    "=aRl$<24JZzF+&msT>^gAPFhaZE{8VxX%`}<>q$V<(d80n*oOkTi_B7Q2lWr+@FX`^a?4hsaK5lkQao~2YCNu`K{EJB-"
    "$|WBh2D_*S(!TsqzJd$o}O4x4&PdU7TN~Bk7ng+`7c!p6}FSZpuRWA%Nx9i52l!9HZ>;7lzH3GJNS$sOw_Y9)AhRnJiO6l9BaG1E"
    "fUw1vTwN3se+CGjIJ<lfmzxHL45=5TJHVoEUlSqE%bX^C?eDk3dEtMtdXnVu?_lbno4vfA0T7swa+$I8ZN6s<s73-"
    "@OY6_(sgu7j^04cFZpChP{=YON{s6J2fwpQS9-96b%P$@&z88jrG#|8ia4+%t0|46h0e>)gA3VNP-OlJy#q&2gH#tSohK96!?WFf"
    "nd3>?@D3Js&cA0QLe(03Lv=HEuof>s*?srmX33q*PZX<nOGq7_ls9L<I{holOQ6WV0%iIs1=iT+jk>mG<%^WqGP<;xep|g0N{TU="
    "IeRH5DPlQ_jx7<vy<-KE56On9_svk3%w0zWSe5x@|3YMTrj~35Z*O@{#Cmpuwdn{-i-aTt-"
    "t<B+49CPh3Ipo4=}He^;$V?A!jda#B4%eE$%;;nx{>fg@jQz)o=*ah`_7=J#J9G(Dlejvy@}nFA}}0Da=D(qko9$b@E~%lc=W{Sg"
    "vgQFg99KJz8CJs7-ke~>U1Gb4#PIV;<oLo5GP0zVS^$+NX2<B2IWPJ<ZvU-"
    ">hoWBr2}1)<FYMHKDOwm=T$lG&ri3Du7KHduS+osPctK?HV&4-rn~Q!i>`kVvu~_{WW&%B^-wIv|It&ZRR6-"
    "&EGv^k2A0!7*~)5a*;GQ!T&A;HLVUJ>=nMsLD4)13)Nf>7#k#cjzU7xVFg4d_>9bnE$4cpp4IaLK@-"
    "+$e%<soU`v&CmW~G*6qs1n*Bx7{Fdw=E8gtW#X+@oqP@VfrQM!^NyGpxj7tjZK0Kz|L^wI;q6>`TXQ1<ewLnyV*ggRGZb74t9Wl0"
    "!K8v?s{=HLZpwZ$n2)tJ<Lkrh$(2BnEUZOP0UJvSTe(dx9sy=bOV|d!`?WhguH@Uf2kF44TYXjy&guB07s;aF*Y($MH@oiwtJI=G"
    "+@VQ<SbrO6=@TU_y#O%kz$UJ(;3eHMlL_-"
    "q(b$b3RGNI%XiVg&!~QPyt^rBIf=YkXoc0wNpBhjwJ7unKE0URSl${7F~2zG(}9b>|#d__#P*xri=ReyY3qXWMSvtv^F<b?M}o}A"
    "NvA@>wV@PSaTH~af9uhK5XdcNTzcbY8lAL7H02xc9(1B7$J#G^7-^XJe@vH^Hmk!K$R^$3~sThT31(kFiTZ9-"
    "wtqP)<Ne;d)>JNOTJi|L~#nS6EwQ))SH|pU&nh*IAvzsfo^Uk0IcwmBSF+hqi)7e>DyY8+&iWTx&?nnPW88uVS5BLiEyn^OhNk3B"
    "snjgfWcOLq-w&4K8iaSZIDRTbY}GsL_2;Yn%M-"
    "U21O<YzAUV_c8A%;$Pl`c$<=}!q^P>DkQcAs{Scc}Shh1g*q6!yb^l!JOad6!5~7#luu~Md9HdfSd0mROcc?0<*sXh7%ms~sWf+B"
    "E*EE!B+Y`!<=e*arZevPpn4Ru#RT)qAu`b7|d$W&*bNUXynfPQ=rZhex(d%tSvXPAf?+!j48m_j^6PbvqP6<#5a%pLBJB{Faqqt3"
    "V+T1k1BQ-"
    "6RD^B&%JIxKBQH6Q`g!?2rXkCww^67oVl9yJhi%E3jZXTmawo$IGs!gSAJ;}euRJsasaq=ET|FXSpii31Gv(6&j7%?kH@e_G~O07"
    "=L(7S9?o+Pk#`V|fjrvw(&@9j)Yq9+Z+ww~o6!7|vr-ex@}b&asej=di#^JWh@>LJZ@#*}J<8<_|vsD^Q5Uwe>k1|K!cpxkhVP%<"
    "qo6iyJO1^M?b?!~QUe+RSL!j`K-H`-"
    "=xs+CJK7?0qZF|@l)u@erwO2HmVZ25MM%>T|isdFvTwQV<96$&_MajJwZC@+ACUp$Cm+Lck@GD%3N{;_p~a4#(=@~b^5N;UG;nV*"
    "#)ht1!pgNH~(p(OXZ^}$5W=;73!$Dm0mOvAz<TKNTL#cNvGwsjMFdNMQPW$xa?<NzL+YjdArJpHciM^$(4(QX?1LU*5!XDgno$0{"
    "r}kMgyT=)LMcxAxNvBAIof5T7J`zN6S<Sd3NB*_Ra-#~bx@qiyR3x0r~qF#?24-rq9kY`(v<%LJ<*W4Euy+>ds`e<$Rz-UaQtThL"
    "4vNN37hyp}E0i4!ZUM;2hPFSj#u8X~$5J>Xaj7a;K40TR7=C3wY2dAf<dY(uGW?Adeq`-"
    "N?ezqfP=Rclh>WA0mn<S5x?j4f|$dL)(oWksY9^GD26@J^ZtHJNLxX>y4Gd&Q)}#}6h&6YNsiV`!hnZ?Y$Zb>PeQ5g(Dzng}f4Sq"
    "2`?862UXrrjT0mmvY&Qqp4!$nOx5g~e(**|4c1W8V$NYw89bSXw5Ns(!f7Gq^gCm=;bMv}6Y9XDpBL8V3Z0dZHetO@ISPJeUi$y}"
    "n^n9&xCe6!$PY=?*fp)djUX6b(VK|Gm@iQ~xY5c8-UvF{^^|6C~!td2;uwdndGHtW8BhfAT64c$f;^?bO;syE=>o>Ul^3Y4{SUX-"
    "T!SKYPR2{(kO0?lB^O7R0LeM<t_MxEk?R{lFS;s%~SCqEz##gre%6IewXl5t=S(L6_E+(!;as;PDlPn{3~*i+ESR)?X1^naV{rcH"
    "{oNZf4$l@<`Wy$wCLig)sDPdsnFKX9CT05|u6<U>05kxMNyDuYbg}XE64A6H9PqS7fS&3OSN@E?_*Mdz`eUf8nM^&NbHUBNHNot7"
    "~+nf-k&%09<=>$gF5kM{QVse@)QEZo`~`_t_4Npwafgj7ib!)m1iH(~xCv4ZjMG=QaCc^M5AWG6}WRA;yvSB&G^Ia;W7-?m?-"
    "(_O*3-en~4yJmWW%L_wM`c?{uClN4R!P$zxR&nxk_)`!}7+w#C>+p*4M?UK-_US5C4C1Qd%M3QbVCg12yX?n?4`e!G5hjTm}0q2h"
    "}qCGh1?h{s3J9sAymBPM`rCO^?hH8?3Oh}o<7E%8ez3o|$0J~;?!=BCe);<%Mb<-"
    "awK)Z<R3Jf(Q+1>xNaEfx{&b0gOx(nwalxIVvVnb~K39xKtv~#x5M(9aK^%gHRFaa`xS-1Zge5uHSgWE>FLW5UZO7Wt$s%1-"
    "@OaP*Q(q1KAveyojs>y#lo^AyIsw}w)w2!}$iDzn!H~M<}eZ23e<VmvD{%*kk2qCEIjsgC`h8xUzLY5I0AR)0XSQ@Wd#(SiP1pca"
    "hNBW2cqF_jM^Hm!A54KCqi%K5cL=S<X6N2Z%Y-lmQh24RJcAm{Ccw@#Ts}^=&!afTpaF->od)H>s5r;5G+4l5ehrIPIo%t-"
    "_w>>Zfm&lv&*Ab72x8DN&Y&OXEe}A7ZaLvUz{uAY8689!A@tGtj-X(z5EyJ<d7JVPj2C#36z4fbaAe-Anf=H+5g#djAXx*W6#&hJ"
    "b>^pC{dvnfxrR)x(bT$5c?>H*+0=H<XFPu2H=p;S!bWzVz&@_c=ouDDTm;tIt>n*9pScDNrCopjO9I$srgFlmmTSUeqFNcNo*jko"
    "4%SF#uTefFu@?B37T)(?v4$CXiN=_Zh;e9^aJ(0s9<8<sZ#r*fQB$RQjnP~De?lfnB`9+TdxQ|s)TUomh9_*BLkP!a=C0l#1P(Sy"
    "qihH~QdVQ)r;Lv+l6h;*ZB<=N4;$AKyb(vSi*s%6`Z$#Rpv?&DJ$6*Fy6abXlxJP|9VN!G_)oca-Xc<pHv)#W#8d?EaU#@2Jt^HZ"
    "A;Y!8;^-G6VoPFKau~~@=C0^JRWH4P*djUlS0dE4ow;Juu<||x^@%}v4K-t~XeNWWi8g?|D*)iKfgbI>;*ZW-"
    "#32o37upo?I<A<$)=m(!@a@P7Ky=lH!c}ARHM}OUWzz{lXOW-b-(19GYb|FJwHmA9lEO_r0xUP33oXP|-"
    "z3b=@A|YUN)i{*udyk0LS)-"
    "(=lGTjMv3x}Cu(vc+cTOBEwf^mCH{qMmaDU%5JS7c*x)(I3{=l%islK&3)y@(S*wdIySAvJj%SH+nV)!9mmgIkZpMuYGI8ZxaOG5"
    "ct;}8V(?)yg_sO&tkFX0dv5APkR$qypz!}HgxLc3#Z6Umi#OQd7LKWw28bf=$<^3@dP&6c4nYG<lTKu5C=MG7)C3s*mGeGFGxfq1"
    "F!N-"
    "ko`ba>KYbI4VDmXF5tkw31(Y=kPrO|f)5j<+e^|5qSi=PoX9^;&q4S7U(LXd(LI$9)>m%zsV2qAt3&bw+64jqTUvVrT^sL<Q%1ht"
    "du@nOv3HRTccFucz~Y^L>}z6x;I$>t1!|`wju|2}zuD^K4RHSf<%k>+H|LF?}DJ^YMsR7>uUaM0=psi2Z7?7MWSx?I~*p2SUsF6C"
    "FK2nfe+VhYjUZJ;xqey6QmJI-^udhQx@6-"
    "ncd7mpDAR8`sZ90EbQVV4`U%TRV}UOtN@U?A3ZP|9kF|L^^fIs|gJJ8cE;w!(Vty!#O<*wBZz{p|t3rZRm9_J@b)rH(bD$nMXiu#"
    "@WG^TT(U+NifKJ%(O+kFaFzVqBHC*FuzBa+l(7(#KVjo>;F+zEJ>RPWtLHYfO;F##+kgtgvF0kI0U3)%LJuIn&>hbx+Tr|FzziKH"
    "NYhGTgjolbLy8Xs_Q!Pt<MgAkTXfJ`=A%9UY6!{4SF&;x@48-"
    "nP{S82H;P^AOL^fer@^o9JvPNTQBuOQV6~Et7qS8X4j(RB(T?PVC|6OBpZEyRe&x{H#1yKyzvii;sr1RMgaYOilZ~a@Xm2Gj>dB3"
    "*4>gTyN|s@Xx<Y>33XxZiG_w<o;%?_TZy5!m_@)d$oEILWktd4SF-"
    ")Ba&Of_VwCt{d#r5efFXD#sk7x2_8HE}T^HCh9&=UWQL3Y78MP;{!?^~mcD2DL`JBQlJ#$rxA=YO5CCF{<v{Nc`NC`vmJ`Lri_^_"
    "OxGliwg3{j71rBp?EA}kQru;ALN0yBG4TAJIP6P)EX1N=giyWZZWU{sW1<!CyH<xP%KwEb4suL+IVZ4cZzdq2qr9)x&Tp2#-"
    "+P`xjKum$GU9N)0Xc8b%pmz{tihcE7xxHkGesbK#38Uf)Tx-"
    "6ya%zZs;uV=Pwuai_@kXn)&dL=2dZd&0@U0SljTb6m4eG~70BI2U2rEx#^QE5Hlm`ma|q?*L1J6KPRYlq_=@Sq-Lu8LMwYO%ftm2"
    "93pwNDFU>K#QmE{LXl(ntJcd%bHQ7&Xq5qEKTTaRf_>=j17khs`R7bL)w;b|l1UQ}p6pZbEKHWM}en_3Xn0H4$B>zN^;?$#2J|%e"
    "eQsN}v~k))T<#)(D&>O->@a@B`k7u^_~|GT%A6%w9toi-Oa%;+kD62D&aHA%X*~F@qx0kL9}y!8OsMy4Os{yG{>(0za7zyT4pFua"
    "VUnzS!1D13>;y#2N@m>!kI#l5S^4U;a-"
    "8J(HCsCunB46+WTd@z8e2>{{LqE0m&>83#wn7&7(RC0>k02HAR+sy6lRXQF|(nID>LaLCeoeE<8|P8v3`ekC@k8ZdDQbVl~mHKDf"
    "Z5DkdRnAvfrqX+f&5uJMn_s~`nsmPtJi#!}6@4_7xLx&)PxSZL$a;aCKUWQl+I$bMCR;Mb6p3dK^TdPaE*sMd64?eT<GJnzkbdk$"
    "ok&!p0_ItKM#H#zIusolsQ}nLZCI!6h>nXpgbJbz7c59i=)LL^1%6Och@(~z*B0L!Glg~?k>=qg1%2|dgC$SG@LOp^bwMjUd5QSH"
    "aF0v<tw2RO+H-x}b^t0MF8GqP8Klae0O-"
    "$Jl46OsKER5RYjNbok(I)<U|7G#awZ6mzne^aE3Rz*30H&6x{~B$zu`^KpTQz*IOrMa@Y6+3FSY=>tdO87C!Z3v=+y$dwqC4;!l!"
    ")(WyCqHNn$Y>RNiI+DbDM`U4|Q9mSj6q%Bv~{M-cr$Rt`T24J$kP!9yRkiPW>%P*7ppY>24Os6NgL11l6g+wMs`PC-i)pAblm6-"
    "2`Wu!M>w_QqXeFpVP9pHB#j#Tf8x#d^HL#W^yja2pdDC4S&>#H3kVWJ-"
    "|36P&byU92q<zP5g>tJ^q!?__?Q%u?|4ElX*NwvIF{!^$!TH9A^#*<k!`5=DXg;0I#Zk<6O9&X9yh_(_s7Y#rGJD;o6EnqWiWp7{"
    "l5Nem{8gzO&eZj#+vDSnL{wYUrrc#65K|9#B2={?D>5nOgEBj1{-3?L6}yHtO6O?XqK~C$s%~-$^&-D_U7mIC$of+>5egGM4qm-"
    "n$rQunpt=gLaE=BS}R=-Rru5JqjK7yQ4?Ktu1==nY%J$>q(tNjDzq%yK{C$zJ%HGl?-6Rg`i19{8E8B6jr@1axA3LqJG?K`)B~B-"
    "`EIHtM0S1H|0ylU`g?anvAd_g+GQf_g#fs=BiXbO)m-wvgAj5r3N^^&?#_J$irGd>JLBfgg3U+<EXS;7^<zqXx~b<H#B^E%tr-"
    "$Wv#wv+0?ZmWJD5<*}~gz0KY2%0Vw_&@8p|g43)v#rAG3z)2#R{*Wvn5CQ$2{5_apA(Jgu^^wD$N^LOt7V(##!eI8n_y8qS2=hE>"
    "MYj{+@C%o>$im=Y%IpE?q*{f{vMmn>&0VPnK3C3JgMHbl^#r(=@V?0ICf{-"
    "P%Z*_ro5<*4PYI4=~i|x)&Vx2)(TBjbCp>LZdzPTC}{3!0IG{Q}KQs<(GuFmLx>-cch$$ge;2>hkyP>w=Xo{@GRn=$rSS0nRQQs-"
    ")!rzl`j^K*{+_3MzZK!eV!zzywlmdYVh{QgXvvC|pJ2v9$f3h9lMk0-bq_yCBQ6N7Ahnim-+Dz*5PpQp8gF&!sD?<_H|8X3pe-"
    "~T-`ZwCl3Z)ABg!tVDs-"
    "+IKW5Vfyu>^YrWsbXvp3gNOOu;L|sQ`PtSK(k}+6_n69dk%==_$xR~mQsvcHIYnWEl(HpgjSS~_?_Gctrvn&?uSuW*jGu(umBH&y"
    "^B?$kOw!J7{VX0ieC?<%|yO?LKjY8a6FUu(=w1tS#AbKYWK7h0DIsH;ToxBSOx6&Xe!FO(kox#%xUz`y92O;&R~YIT)StZLA?-"
    "eZr+ta3zFPW<w~pE%KoksPzxl^YGeZe2_60nTANxdln^c^EkU;>fM9!z)gL~uqw+Q)+<5s-dm1IO>P;|)w!}_XuSFTP-FaX=T%*V"
    "^FT<o+*5xOl0I~Q^9&`U0vEh|oi6Yn|uAMZKGl^rSwUct}MJz$K&aimr_gQ|e2y5P-"
    "QgR8_B^HAAcA57rp>4Y<YQVNtn`Qj7wltng;f_@KyK(poo3^L1$H&hgxPL_fj^6`&#WYj}{F$`hC(^Q|DPut#t6Eb}chMXVbKEXR"
    "9;9nlT~Sl!NjYn^#80K>+{hH_x6yhfi;-"
    "{l`)@j{a;=2TCo><(ZcXgffKixYL39};h{jNYmFn2G8Tu91>m?+pS|stHftjd9n%8EW<?;+7qSb+0f;qL@rGG^+qZdG$qg$CW1fg"
    ")#bT0isM&`&<ea#@3oYpk;EM`q=S~3amzhUdWH&iDHu095srdl~CIDy}5PeM(19d7(9&(U)IxyJX%*z97T>0l$s*b@O}$Z$=1g8#"
    "#aa?L&ga45M_-"
    ">=v<H;3!1H_;q_f73oFctiM8zV}{60v~BI2`rKBabiJK>^}QjDKPVJwFV}nVZ(rKFWanSjh@#~(wC5OPnF?XaU8nUCG0D=cT@+pe"
    "nOh}y)Tg-4LA~Dklr%c4YfgdOj|C?u1gDC|1copcnmW7>aCk!Gd9%s83~&X?TWFlIj6cA+d4Jg{?s+)74U4e5uxfc>dBy!Qv6>-"
    "RlK4qW1w2+-"
    "lko52PZ&wd%6t~c9Jj{6YNGn<nj{P!6|e$Uf7)D`>UiY?Q`sfj`c)=9%*)C+Dt)Qbo@y@?bnX#7*vC68S7KPU|jvQACx|Z%wB0Ag"
    "1QZ#xKc!>TZ3Gk_xyd-"
    "M@qU7pn8|3w%h~CYvpMuuf|JH{?~hJJeY+eZk?lK<Fph=#Txytlb#|)hWNvHYpv}8rS~4j#j(a~+2=oEg+GFSv`P|ja0zf%+n4ov"
    "$zvoRk_b2f?rEY0APY^e$h?8xT%m`s$_$!S(FOAtuRD7En5C$PJ&;7!_sr{4`cdr4J*}t3#>v8#!si@9LRbWkX8Qn_t6CCB<uC!D"
    "w0?fXyt9z)n&@|8i=q<qefV=H3rQ1CuH3WB-q5NqHh0R=o`uy3WL{>Hd^Cs73C0%99}nSDO@g!C*e)93@-"
    "|pFGK#3Vy{Ua$M3W|9$kK8d0vW~2K~$|6>e=3@E0pi3AGO`6R?Q68eEC*|wS>ZSAfEI}Gs-"
    "g7VlKTI#7+fgYWRFb*FK?LOC{Rf!OlD;j%6ec^je?)<Y2BTJCDC_@_M>CQUxcH-le%+y*_;s6%}jQFDdc}E*dlMh}5y9nV<@59G3"
    "!zX6PS8!}}}+Fx)3A!QR)Yc9hj4daEQb8zFTkWxES1BkTA&egzC=1Ciph)xl&8ap^qi0oKxgB4RtIeK~T#ODIk27f4vbOB;!UZfe"
    "rS3enx#7ID8)c8Niv-5Yj5Ci`%_<Kjyac)EB7ic%-"
    "!qVb(2XMU(Q_EBL$gNRmXY3%NLKp(m1BQtEZzFo5kUCT)TEESY~<=0lJ`@PtK4%z3oV(Z3BUz52MBSIEpxrKnK5@*x%aE?)9=@pu"
    "x#}Q5d?TfD~IeX<VFfmyZHPQpRExR<MjrIM#-P~-p9V$R-"
    "y&f$d`4g?D5{T31z)KCnhX+{ubSL9Ji^NcuT?xcS5u}_%qs#c>U8`f7Ym}u2^o+fF_;Jx>ZdvcINO9XNj!A2R!e2UJ+)99(GKkmN"
    "B39o^Ysll?aNjeyE5}l9raq`#hVnEVOk0WjyfqIEU|8hCtET=j9Wxa)h3wrw2}wnnAJsXo1a9$rQb3=}%q7E;<Df;ZHAn1V3~z7<"
    "%qMkNM}FxEx7e2_#*m&Xx^53ay*hU;8%$I~vb!Zti9lYH-gNfAcSg;3Cfl{m{?Kz!fyov=CT4d{{OorgD(h$BXLleO>iYj1yVq8|"
    "vMameU&ZKELgeawztp=xNYfBP2uTP8Q^=}Qwqn%<RoNyn{_feoG3Pp}ZkIzBv|Bnlw`V_luQk`4WBf-#H|q-"
    "B_7;09sX=!>btwwJc-@^PDG-"
    "b##PFBft2pz?!32S(4X+haISlE`mc2D(`KVot;4kKf^7)u>Ocgg&rrjy}l55MZF}!|O5R0a+%{*rJI0Q>o8=7e)MKdCcR3kyeErw"
    "4LDtGLzPu9L$nw;P`scrUN<Fv*OR6^D1aNsCs2*b_CyT7VoaZ2tq>279F%tZqlI(MCRWd}Jjz3rQfR6p2uJI`HP-"
    "$~G)F}pOU^4Ger0^T`!BC7WcT6voCa<aF|l_|sTt^2_1zYimdUQ^cfsWVi0QuHgtyIc21(dQ-R)OJ-"
    "_Yp}P(S{<!SJ#Pw|<;XsB&K)2U@eZ-Zf&kZj{Hy9DoTlfsR}o2%e8+Z5Fx9*i6Z1V??)&Ijt}{f^)o!TeSBNKZ^<}P4<XBknne0b"
    "X)Ny%ljh3|pM;>3VX4)rpNfZU!+;XL<@I6a<N~A;`>7hv%-Tbl@StpT>Q*;~4*8FcLnPC6JooI<Rpi%gINy;gKXy-AV2uk+hmL{#"
    "%q7&Ti?iho``n*P8uT^;TXibm_f={I?)tVjvJ&s!#V0BD_#1&@RB+3%7dlJ`1!*(T(rKEMS%}H<Uh^K@4?$*AzO5e$qYpBQly^ZE"
    "FmEtkQo<dsk8NQnYHQ#o<i|K=I>&XdfZyrJO>YTMx6Ekhe+b<3%Gp-"
    "Ch2U^bDgP6ILl*9guu9QL2h!#jx5C}tBPMwdKNOSI~MatItIiJE-uY+|Bt6U;~{e1WDU@gM6*H~r_dZ@<kAkGp0I7q$bG>D^dbW#"
    "#^at^mfw5K^s7fcuj|9}@0#Wl6$O)7atL;Cnd&@yEtJoYH2+0skqHRr~V?9m>qL(E;K+G~z!y`%&mZxGv}DzYTafjgA`<~2E&kU_"
    "@?R7~<IL^X5ID~U-%Z4kK8y#(!Yo-Jb{=F%HQmVS-"
    "M<UT)Rbe^lzy95bs;|XoCj)znd$|J)9ZI%HqfXt$+FqIw!?Q*9f5Nz0MJUlv&1_1ioF)q2Bcxxkx_uTgNC0QBNga#G!NbNMTaqXi"
    "1JQ|*$!?z~?<u&eBmqe{$^SZ{=jz%;J6}{<eRI3JNihkE=8sftv#$pcvZE?LjX*n2sQJVLbdsS!FiOlJ%Rsu#kaUq*CRK-"
    "Z1iE+?8-lD?lZYy5+oPe0Kl27PUV*k6HxQUC03CVh8c4FvNhZU*88K`*2wk8IglwvxJ&#%_4-"
    "pq}ZBf&1s9ejSK0V_X;dSf5={;u}`p)++`C!S12za5e*q??v<hOR(8&e3R=OdDb<j1X*MLw0OQx-BO5OH<>~^LH?Dkbo+tOw(*xF"
    "6Q01le*sWlI_6TD>RtUM)MGJrN_ST0dB$EA+4R1rsA}{M8sR#TEnD_UHG*+bA=Ml1bq(}ZA38Odj+?$RZ(m811N;Pd{(YHYI7Xb1"
    "~kJmwGa&iQf>z*N8fyb@Lxe??BigvCV=u1=9H;gaopAR32$T*>#2hFzW{JGREZ|1kp69n4ik5c0nrCMGZ{=+vFoeVPEHVactZ1Fe"
    "3;luV;$KKtkwY06kxQ<1rER0-0()N^GHMlBXAwa?hE%vo2ZEPCYo*!-G}DqxU_>mWSp7s?`yl-ElRAgswgp)YtW?j$Pxm@)l~`i3"
    "G$~$wYQ*H`vZw@I%Q$gG|nM`a~(?x#K9)UFx1he&)by5!zF!aSzhB<`Stt=1oi>BWmsDRTBm1Qn>@TnWL2)NwL=y(1w+W^dKyhC="
    "$=D^I;`SB3qF&pqtvlubOfKlOwK7x5*?$yPXXr!Yf<K47$rM}@$l7`(z%^<!rs3R<BnoT3*{!Rs}k+VEPDD(f33rvL$T4aZ(Z%^J"
    "(~~JX41z;B61|#@a1QreJcV-Fs^z`i8H8lIYLDoqdb|$N@KuJo_i#+cvJ{#KjH{sD1+x{bZgd=u-"
    "Iz2Psu}cyuGX9am9_?=87j6$tqM%yNpX{raINEIWf)Q0t<HUhf^ZLk(4f!l96wN7qTVV+A*f~N}zx_S#s!iC5C0s(@LThIvd?V{Z"
    "rxCdG-AqQ_6(X1|;6h24R|H(O&(vj062~g&xdV<z!apB7r1(;Oc@-"
    "T12BtU6N>^9^ShE<id3!6D!y)(KQD_&|iRoqeELjn1riRinKzL7KEnsNWs_$_H<H{D-"
    "EVr6}Jp1{hUNaH)DEEz}FGRksj9w(ojaJ>`g`XG0kS3%VBHI1Jaon^v{wZaow(QuvW%Nzm0Ua<c7((Oz4cd?HCQ&j19$+@BOho;l"
    "^pkY%OfH5=(}{MW|y&x%zY|(qXQ(+)eb{{wus%V6IhzDTAHHBg|h%nwO>Sd55W*I%Zl%AntM{F&3hsvXR}!k|64UIfbtdrye|J)6"
    "xpn1Fn7aj_97$J2Xe%sz@|r*C9*%<~`(VC?Q9gdl`P3JLI<j(@H8uZFB^(v^LZ^x+sGYXw5K!0cacH%4T)s2_F1=R9crI+7*HFa_"
    "_@2dLO~%*>J9rjW=c3xD%>eWz5g9_pE@IL^KApJ*N(UpW`zc)+n3Zb2N*iu~)2B{<FzGw+K0AwS^<hid=3liaL$HXr3I;)(&iU&_"
    "p&d95=gZ=r83LQ=#PI+V?e7sJXYbS>(Btw+_*E)<I`$_D2TQ+@;BLs~8f~nzXz+sA>Weh(OY!HN4_V0Ru77rc)~}rv!mqIi1~O%F"
    "gqZd}dwbw!X$t7cQAuKs%+@(5aKe=@8Z`Z8^vX9;o*2ala<PMpFS7AP}n54M$ofi6np+Zl&}Q04<8=(eP@1Y*QKKR`(RR`mRK>*g"
    "lSK3esQ5_~N*3?b;mGqhKz>{OcTCrz1x-"
    "J9>8{Tp)qya=6z)EKQ=0x@fI4Y`^_sT$@5_iTBhHHii!z{l3H>N6wt(N7}ovp%8*K)Y#V;ETtd|&bg+&d-"
    "mJ~4)`MUIf6E*L)g5gbz=@Vb#zuR$01}WygI&Y!tdJDCt`XjK<n6jiGee06N+$;oFStFBX!cb(o_AqMc%~d0Q2WL;&Jx#CfxcNF?"
    "Ef%APH~lUUmMK^yMk*-"
    "dW@!{3GG$t1PO0h&g#vwO0dJTI$GiN4xRde~znN8qI0em_8_*N@$zR>wwo}8TVCN)uGBdO4qJ;?fTdrbfdI{4SO`+m;zI|ie?2ct"
    "JVQ5ziN+7bZ}q~EK6;&U>Qchp#mMe2byx-`WuJ!YLGkw2StW5t+s@-0<Wxj%EA$DwAYa_%YBQ=;n<Fdt6-"
    "6;*`uW&i7r9?(HASsvB0rC^sV$hrr5%1o!1g|0vaSF#zs+nBHanu(CnR5$PMLs%7>$rtM}a&&rBoLp~;Brr~)*WLQ%Fm<_MGwdaP"
    "0)M|#m28*3CJ{YX;p8XlfHlzTTPVuD0}F%@^p>Q7w~*+cWVE^W|o_Kd*C(>p9OYa#;#I$4x*XI`XfYY0dK85~$qK(so|hPru$H*?"
    "<->g`@JYHHD-"
    "t$9GIV?m#LdWMcW@SPDmpFEB^jF)aMm_!7j=D}J!`bZOj#N{qkeWU2MwA&Vah4pjh!yYHkTD5o4ND_i<fAf^J*^wdT$~=+}u18F2"
    "X(Nkd(bx8``6l}9ijlOm@H*Oa)U3zWc8w)N=XCEy9d)h~dZx^egS)Ju8%HvG4pnY9h&`^b24{wIrHfso#=Da*plj7hD$aUNnj*Fv"
    "?d1K?-"
    "42Iref45m2KJCuk3LpNk01a@ynB5^Pu&(!kG;0C1I#aZqtMLfs=?<1kY4nihjU*GeUgGfNtHUfIC43j?SApm8pS|m;oO;x=6!@Y&"
    "v#Ot#NVPK4=Fs&P`F)N5nYvi_(fC9CL`3-"
    "t61Z}H6OcLN0ETJGuvbbcDVJoAdM4vwiebd=Y;36Jj2Rv_2ZSYQE?MIe;yZW)0uH{P_!L*$FKyl6XjOJrrnp02nS)~s}AS0eEJvx"
    "BG(;9cG&o>v({Q;&^=-$tm&Y^!o@zrX?+Ysy)A~Qh=5`3(-AX5-hxC*v@DiC_KvsfSO+MiI6GT-SKWU?J|L;Og7uvne|LLwp3CEG"
    "_gwP9gP388sdg$%y!vjiLyD%Rji(A`6OHO^E2)vf4NfE+hq*_X1bm814C2!D3|gut86_Kt>mXgH!4>V;Ry`#G%tb|`Tefy+c+k1("
    "0k`}p==DSOWu-)na*iQfw8ou5N>{*6ebAq<mL2J8H-(>ks<(9v&oW7Kc6}b!Ht2iLJzhXJP9IIxxJ-"
    "M56)S{#2!gtdG^V^gy3Kh8kU^>~#HN}gDPL`-+GJBdGeoyG6kM*MyEUIxf}p+JBRZLhTU|LvrqT{f1D-"
    "js$u$DuLgd5*NP%$;wC%PO4<zl&b0GPvA|VBbo9Hq<nGMPONKVF(JD;NHNS@m$%{QQ$Pqf6=t&g}%pkpxXUh^2I9E`Ut0Vq&)h(h"
    "!7gvpeeTTHMAQ>X_*1id$Pk$j#GiKn2W?4)VZMsQbk@80@Iz)8vneA~YdjTi!y42)~ErMNmYw$^26Ae<q|$K3VRG)=H>5-"
    "6u^NLMY8?$SKjJ2Y;;vB|!?qW8OcSbaL+aUyY|ydLr_qf0_HIRi`jbZu$qllY$XqNCEBqnultFJ?3m<z}$Z^&n^pDVEr*@1>82^x"
    "Cn+aS+eLfUeo#xTFAh4Plv*t^GXWfy#__p$b)(M>fyV)7&IuL_#m^Yj5&l_Hrkxv)gO4_?9bX$vM$f<BM4a98m`6A{w_Xn0vQOA7"
    "ZtK0(S6_-TP+^!@R|2fmz#iw_b^+1cgXL!jEL8O-sDXQ-gloq|!VC4IW%vR{G7v>oV8qftMPCR7X2&9Q{}u?d&1AH)>)@&gNZ=$A"
    "(I9HS2R5gq>6O1)pz3S5BmN_u%EQ?F1!In)tDXt4osY3xP3G$@OkD5d8AB?j;*Sr}B<KxPUR&q@@oQbC+zHwh$j?u2o@(j5Qh=h}"
    "GBk2=72_!K*fuug*7$9Co5pFOFyqyP+3T{ibue7QHDEdmh5qdr!r2yDT=YMzU`?c;!rE`x;r2Yj(R8#c8b!8pPo^9gIKja|AgUBT"
    "_fX9w|6l=0}375?j>A$Pl&Wm|&#rl(3g=*oY!WldWcKJhTdB?eXS5SwQz#K5s{=8vG)h*p!#_XhsHXLU1uDmsV5(f~mxp(5RZNVr"
    "sx-pl3ah77HF{7N|9N+9rx2#2;bYK0mRUdc5kku|5%3QiwlJNrt<wh6roe>mgdhdn7~`W7ftBdbtBT%JB^F1LA9Gnv%xe6_zfSUK"
    "4gtjV4~6JN%^nB~6O7T&U>#sXP;?#r?g)FL%Gun?U4J%FAj@1h^W(O;r0O&$(lWnS)=y8+=5CE_O;~+`}kQGa2qz{Ed_@rUuaDIU"
    "1_L1YPtM7)-"
    "Y+cb)3PDAs`8yGeT001n_I5cao)GbRkMoS0t@K|~mWWl1dh0m}i&zUS$>wt?TnwbPxYZdf`K_=u6jveDFI7^EClH$ju#PN-HfMZM"
    "eeDA%`|S{rwX0=B!bqMxq3ZBjD^HS<bNC8ZF##fZpAhFkzRuZx|HE0#fwcpbTvr#9cXROHi+fS=^4o5^SpbZoP%-"
    "6J}4A@)O7_FQp#Y%tnB9_r!*62ly043jkX5K_alxoef?X1Wrgn||6z$Ox$QNT_ep)2eE8x#uYFUNd((GSOfr5lrPr@|qnU2sx{u-"
    "y-_Ipnv0nty7=gjVz`!C(ff%)!f3E{}F!dugxYmLDV5@*`B!8VzFcCIPT6nv*T;SPMLz1@Qo_qz!;;-"
    "&r`V_8Gl+s(H(p##N`zh`VuAL$RTFrStHpk-n+b*HCI(-"
    "d3Wp|v=!vQYyU6)f43|F%yf<vVfEHtv$eJPRF@)DEeDQqW|_70$7ePtexm0{YmBw-NAGrdq8%MikGbdk=;%hAeVpbq9j~c1Z%iOt"
    "dTtfz)-"
    "j)c<Q*=;)wRhIqCZDUus6NQ`u1Bke`n=>It)X8CfJQo#JDQCibdZpeA}FTR!3&MqZ%HuZ)g!Mue8A{BXIe@qY<XDGeuB9)Oe4q`@"
    "0k$D(j9AIimbc0;5Ol=(D%Ou;s#?U(mBTsv@sY=V~B17AivdPic>~xCg$|-"
    "Dw8DywcIm@TH_ldN^86FjrXv7SenOZxZ5%lGUKE#3y%Xvqz@2eLe)moC=S5=~Z@jfzFo^^2nUCUBJ%|5aW~F^0p~jcFf)~o_QT$S"
    "vQ~s%n^oa8zuMN65nzz14V)&V+285*tG;7ZQ50q*)~R8*y~uJ?vC>{E9ee;rD7KQK3$AfDfDm8PHiWe%+g6HgA14r%DNp%WLy(HI"
    "o6Y8sj5PuubSNP+zw|JTvE<Dh8zw%r}`$?t{x&OknIb})u=A)72(vH+=le$9C3WjgoZmeNMCSF<gU+sK_@o~hb<mTwm1oyBh1Pk8"
    "t0+lr4~M_Mea1Ex1n8<RO&QM8{T10rZ*JM-jZP!&j}c-B0*)12hzSm&_Cl|9=JdI(@JI{GorQXbV6Dd-p6fdu{m)a|ChKcyG-"
    "9Xq3(1=j9_|Ns5}tXwWk6o(QFd;wzHD~dPIU4qGe*PyY}5YC!OVDLOstG1emz(5UTTije#il_m(V`SE0Krq0PuspUYIUZH(UZ2bC"
    "vPeJ>cmYHa^&KOPaKH*|)!l`(b8F%@KVN}i`d!Xc}eGDk)t>=lt=DG>DR=PfH<9ixRjlT8i-"
    "^lqZL1`vWcSFKk4bXnl&HN7C}Y1MXfjBdi38e^(i!Q7uX#}BADR?Bwlhd02p?$A{FhJa9#v$?-"
    "oOfk|;GswrTqkIWb>c)>=u`_SmShr>>ppNg3S1BGW&KCk)KXXDog2C7mh%h1-)W9U{Jgr-+-"
    ";IkY@r@w1t*Pze<@Ev*yIW_`Ih4YT156lRlX)+<&kniPH&F5-"
    "?OjW#K((IX)oW!vPy5zMbCHVav}qIVwWb(CaNV%RAh32^Ag3KwTI~b9k9%J4a7c)5YTs4P=K0Qft~d}TacMNf=U)y;?TRh3pk*M)"
    "VSF?@suXM0TLS2Wb5K{+j73Ql9@&v@a(6+ekcDX6jdYfAo5{#d7fvbK_9T!%LbAlobuA^&oHT&z|7zh|at8=fc!eoNUiZm4tS%g#"
    "t6d0s_fHAu^6>Jut1yW}`h?MCR+(f<FTF|OwwAyXu#UFQ31ls|)Ex3XZG|z-"
    "Fh$5TG~0CVt;3#q#Rt2FsY4hsmUmqheOIT~^pD|85+no4UJPdkp%^C5^o0v|5VN>WCL&|$x=!5DBkI4k_4&$Cycw2A&Vq`DQ#9sW"
    "MtzvBD&m~%dUV%mBqc3r%Ccw#QrUj0L$8#}>4=xR)o~i9q>k#rt{{xWqi5^vRoPw129Q3p2CdYY28_G3@>3#d%jbQU2h8|>9A@MH"
    "kB~^<YN^^ae54|TZc+)PbKS2~4(t6!se$IySEO(?c+nZOH_k!wl=1VaiY-nv3mYzFpHIUtmXEX^oe4>g$&>5kG57S0M2&VhhJ!XG"
    "DXHz(^6Tc(11?ZcDi)7FE7<5bMLY+Vb`|+%PL8wHVH3x94jt{YU@$G3wDnXnGY8uPCeq#BE;*oQpu`eru9o>tx_~zY!%<@&trK;G"
    "HGpNePFu-"
    "f4ER{e_x_qjW3klwJ8+L$=v8N4EEbc2VOeyis*T)Row}&dO5a}xuf}YK<4eq{rdV@HzaY0!OQOAShwra0Monl9ldFTC=MXyFC@g*"
    "*r{OsXl~nhzyZ0)8P-~mTHTSOIc><PRH4b~Ff3C0D3QMIFn3hi&9C1To$DG5<x-"
    "QuR4?@V%$2QR2;I^;aGs9+Tc(hMx_@Jtk$5q2kSP!q{*%N^T8(<L`&bA65Q6O9LLPIbv4+M>j8`jZ#dBqoFg0I7+Q3&C~XmLx^cF"
    "^!?Kc}R3?QKr*c3;>Nj*KLGt%ig%wkB*Di8UcpG9tzks(jmh%`LoW754MA)x#NDf3hRik#&_kG8(q@afW|f&nm-ic9(t=7rJ+YJh"
    "8g90a)O0+SV{@N`qpAahK$i?tCo&m-"
    "IuU=yflG`kcU@N~u@obffpM;Yr1lE?w1hD1pwUDNSqeNY583(VPOPjZRLRR!2SRtLFh*FjO)*81_r8NgCm9AsB5lya8S;thJ6V9r"
    "0(Vw<jX{e9hR3Sct$f8LMsHrEX<bnATY2$EnQ<!nRlt%IdH<5ubAc3g=_Accw70&0CjJ-NOGQ4_;DSgAukZZ=gi-"
    ")aliH#0{mFH!*9ag!)9I!X#|?8Y#3%zUV6G2O7_!;i)gUh8nO8aj$1_oePp(P0l<+S4ct2I0az$2>am}InU<tP7zuqV`WLDRqOVW"
    "o{6h2OYXk<4L52yLoAL=dSF)0_495paMO2^e6NJvckX(^<}<fY4$ie1WU+K6yKVXN0O+D$u#VDy1-4cmX@`S^h(}yhr&EW1v&P-"
    "Qc_?i&o{_-EFZ7Bex^kAKUR;M<>xxj_Y4?m-"
    "utzFa;&2bE&f4ZD)#jEhz^tSPiD4yhtfNdf{xpsEQ}s}ZooeYN<U=d&{i<I*lL@{irTcz0_ii~VLjTc#Esb3&Eio_V*=tT{z`B$X"
    "`*h!9iQPB0h_xzhqsArl1V5)bw_6gcwIV3jm%*jvAvde;mPLXXR8xCc2Zfyv%>7I!Nn5}A`y~!Ou840(bZ~ITToF@#$tt}PHyl1d"
    "9QxhXao(mnbgt4Utcu`ZP?r#J?aSR8mg92<;`X=dg1W!hFO|xq#9eEfi1M6dtT_-"
    "Qp0%lzNxh9F#TXNp`)!*#qRxaqwixaCvx}vO{jTiE9pkWv-bOTbL^)8~kasotNG#t?JYo-"
    "?s=Zge4Z=UeV)Azf(fZOUjGZHNqNQ^OuV}U-q^qm#Y?#;xe%|Y-DhAf-"
    "C?Z|A*;_h+2$>WGc8n7Fo3Wgj>xly^8Wb~+awQdGr;qY{qqdJ%(ocrl^0^P;2=_khn4NRPkZhKB;!0kkT#kb}>}XMZ3ODemt<e%j"
    "0L+?O(9NA^o6=SAd@(Wa9xiN!)Mt@b(AU;^dJE+{n8-"
    "FpZ!OL(v5|l8;}=yDfq_YNLtq_>;MuQWwTFuG3cMus+_AM||A?)D6Ny%^&TbSf>MA$Mn7UhUjLfek!}1=pSXKdcA|f74E&?Q87gp"
    "AF?y^V^{nQVXT2L#LG`6s!(Y}T#oX4iLq*rc|dOEUilKz0e*Zob{=+U?nsM^aTi9|vVjF;}d(^UREVXXtmR>#h^GQV=YY@ai@)i#"
    "w_S1pGrSRfs|YmB93wFpDeqa=f>`H*p3v}U_eV!3{}%@YG{UR|3)j+lvZ+$*BIWdK~ATbh$+x64=@EWe<1U2wtxnN)FG&*S_G@fa"
    "JsES;L!FJWlvfI0s5_P)TE6)%MXq8T7M!Kq30*SDeTHm=ZsIVv`=PPHWJ$ytavsk#fK=b7KE_qaSRpV;w?x|gQ-vChdwr$v1p=58"
    "nAAvulTnYh)Z5SVe!@GyZsR5fSlr=%m?oI5u%O!l2Jc<;<fx5>_qph!4M97c3UhGQT!CrzXA8b{=mdo(q*EW}PDzi3Ghs616--DJ"
    "exfh0D-"
    "oDe2lV%yTvh(k*acADXP17~F8X&)qywwYaw70)^1OcVF&_TuSYLH}p<DN(3aqL`R7+J;`D)EIt4C&gO5(O>;sNr%Te6vCRI`m7A4"
    "s6HjdGh)AfuHL4PiT3P*U|ZVL3Bqh9b$Y&=2#F}2HjHKp*GMxacZZAFx$kaQ6^r^A;jp!|b~37fCDjaYSI6pzsKCZ!|DNh5%kdfD"
    "?aZojj@C(fyBbD*jZ$|}Vue;<_i9jt8f?psdul|zCd?!m1P*nsx&3oX#3M(l%IP&K4E(VvkkPYWwb>4Z7hliRsp&}_2BIa6T&qLO"
    "v*McD(j;T|z{BdtVUpsddqXy&t>R9ap5??f6LT3AY|U@f%|%t(Si<hJyoD*4)pTSwQ%Q>5*0-"
    "Ooi{2LFr;<xfv_;rqsY3(IqK+cFpbWLxM3P&Y8P1#4=n7))($Wl0vzoMW2A;P;`{ru19G{8d%X6c%igpjvC?eG$*~l`UYLnnA5eJ"
    "oA?)>_kP&w2Ul-;{$O+Mk{SZ%MCLr%IFbZP6W^Umua7-4{~^v28i8Yy%~5Kq@kScyJM(Cb5;xkYr2*hQD!%V1)y?7$aM^Q-"
    "mN)SVF20Is8vg9N@~Vu`+^V{wx2_B<c7Il~2w4rX`5&e|K9%|f+$ZIt2z;}quBjpMg8-{$%DdnFHY-epvFq&UU1&%Cq}r=Ep2R`g"
    "mdx<Fj2jn8UVi71itXvprJI$0WCK0kdelj9yXhgrIL6;TXC2tXmc!DTy9T;~uirxh=m@*D#75MwktGYIIdWG)sI5X?LeKo_%c*ad"
    "qY=}yJcy}XhDwvO2eY8ny@vd=KF+YJ*kp%|%|DvJeaE+Y8cCp8O$2_Mqm+4zP;>v=R-"
    "n3|?p*+*#~QfT&p5K%IzCy~=tPfXW1)wrHbpQD3<=4o+&y8FP*!g!c)0El%bMLgEVK8JvEjabb<{SUy#bR1qUB+fozyr{OuV1&GJ"
    "1oxoTmHGaf)f}X49=Rf0@W2jAN7m(B9X15BQs1quqP9tRqjL-Ku-^dU4A#}=dM<W<C9MPOJt*q@EyinC4MffsaFbg*-"
    "c42VEjc*gT^z#KO9p)h?!0PLrr}p(O3fGd=xu_W9)pHRU`)35=Ewtw23uUk!dgqBwvyJ<k>kr4aHvR>QhfJ~^5reQlMi;hDrw9O%"
    "r<n7t`hl9;$H7?l6Lc&-WyC574uGl`^Zs)a-`fdQyri-DY)DLYveR7C6y5!5*U5lHdM@MM4vhqjm4kx?Q^$s(G8kil1fCzX|Yu9U"
    "r-"
    "T*d6DAI1O0wxN<c(+4<ec=#+mW}nDJ;;h)XE;#o(c5t~e6bdNQ7ET*pm0o7|VPi)1nn?f!*5y+Ne>)<^gRHnF}E(N5b{95#>Rb5b"
    "$(5o_&PPW>e08jTxB($lmB)<`<3s<Y8_AH4$f+!8BYgGbAX7ruu^(pHdVNkRxjOri<8OqQ)IwfyKIwIES*O*h?C`O0&_-"
    "D1@px{j5_3FFb2`5KQvNWEF%x7Q|Sf*r(h@~6LQM$bS!?(~CNlCo!r!^JWsFTP`K1g*kfrv+FgL|T6kGD^f{B$*zSI>Xc7JT#Xzd"
    ">o~DY2uYfT4HpTOB=Y7-?qnn)YAEYVdD*uOVdFUL}Dg-"
    "H=&7Np`R)v!Xiu0V=BMJA!8*H`PamO^UC_=X1s?8PO@kIB|z?WVZ|}M#6)}ynuF;~a%o~E?*`dZRyn5dJ_#MEC-pot-"
    "I}tk_P|qu@0p@`inj7K7e}5KZ26$#I;%a;{Q5IgVU+y_D@Wo2dOLmU9@TNi40F%V&Ip9Ft&i!wY=x`5YoR-"
    "3Rm}F!;<_gxyPViYvPXi8lmA%HGivHTN2?jr7%%Dt2Z#E$CWgquVI0h5R+&RPl*zUgKfD%3@Ic{e(Gr#qXp0vaB;JG3ex@~{q?{!"
    "8VAeZtX}fRxD@i2HRM0&&ZY}2SEopjeUh6)Bi^;#|Yz*60BN3_y3bSiV&`Aq~+*;Y;l-tYPRHu#T_F0I?$~yQ$$-"
    "E_ldm4!jARG%49g+0#w$ijXe0du>B0?T<Lb!x%^n*(@@f4YaWR5*Oeb|x_wXx)FHOFB$m?|ytSVrKr>3yopwB#CnA!3ja!t`-"
    "#5w=^>Xn<LaLa`<v1~`GHNE#^p@l&avAy4n#`n}GvwrOfm!fB1nR){&t9f-"
    "u5R;@P@$+P5Itm~_g*CS>FT;j!vuGmrjh*Ij@I$lZh&B=S0V0sUecowziC7sP5p5o<9D6cSxQSnZ6T~1I0SjRF5#N|0w!=K$iNz@"
    "&gQ__9+rnnH+(~mnV?VZcD#uI&U{&vbL7OHI<jF29tj<HGJ0L;)suc{h0hP#(Iq?@$XZ-N$_ov8z<mv}L+1WH~<OKhD-"
    "+^bVYej|t(cmS{GB&o{nq%|bYltQJX)M*8|A>tCa&Q=A(W;jpHsYpg;Y_MS5mkE7YJYx=G(sYd4>PQrlf<}ace=Q9;-"
    "N~1qBigExNxP*IM5kU2aq9LOMQ)+r)JjDph}YsE^onSkiphO!m7`q~s-&~dVFIjW5?V72+T4cQJ#x7=r2f=|cN&Vl;~F)Hl-"
    "5pJ+fGf~<}k;4(8pR*{&kPlx|20cR*6D%w%4HvfpyJ(sJnBwl>i%(mXcz6V8HDDTjD(AKR45qby>!o7WO@c)Pzk6O|f7qmTu%+OC"
    "0D@H@9U_rBrsxmMv(RWn>aOufl73Hu&DLk7}~Y5If33qnUBCri-"
    "_@3Yk)lfrRLou=jeptCX%JnmBUxeBJv(9x?ZA64ufkDZ7?oyKHr|x17g}i|bsP(raguMOz-XF|zcQMh7o?b=k8M!xTObext_7#Yn"
    "IaZRde^z#*1&1jBzVL;vH7nbuO$9aW6d+v-"
    "S|;vM6Vu!Px?7^z{PDXtIB?uF^H4})Qn=s>{uWlQLfUh2}ehpBLNVSg>*Om?olgpSCk?qp)NusBpMif|)j<Jl2##{i4|)e@W>Fm*"
    "W0v71d>0mPSyyQJB3=+w#YpO=veHY0%cWyM6hByMK7Oue)X@=EV_ivDb=aqFIFctF6nH_sbxm6{J~xnthuG;O!g4>*lVps~-"
    "TSVEOq2Y@Vk{oPrCETI~|B+Ppb*0huLgvr#wuZZ6KsBH)HQ$X!)Y~yq|Ts?b$C+@k97{9%@Ibhc_+S_e*8R?|aENAi3=(u_IWA&|"
    ";v8g88GZ0%L3#Fc6<X%_kJl!faW}A$E<&ZZH`U+E?c=66w__p9<SWh;&sdG){nim|j*Aul!xvfI;#aDGUX=ak#NM;!Jq>*o_D8G{"
    "tu*81cnIDu&&wcCcGQw{m0!XB2F(Wb^(T9$FVr92&3=U&Ce}0USpmR(!CqNaJ?CJZ|JWsRIB}Bds;fy&_OE65W0Ru^<PO_HV(68f"
    "{DaYR7;;we!m0U+d3QfKqrNaTSrVDjCM_-jB;m&#%rc2e0*35v?gk`Zj4K&^J68FaPPh<6zb;LgCy&Mh2Gp%S!ac0hrrO-"
    "*Upwo?=Wg3bik>HcR<!l|Pcfq>3q+vY9n{sUa88LMVlfl4kSB#aAXbCki>TQG5s=3rCC<vgR+fYy!RV?wk&krw=x4}xXd88YR4z9"
    "U|{?43?ml}itKk6Qb<5DM6&=4S==jskzL&aoM-Q6i7iX9{d&CP7syc`X;<uZdoKv-"
    "_+OF%EE3ZmZcae7nZC8bwl!g00RWieB=)R3{3Xv1c`5gmJarzRtD-"
    "7iocz@H^7Atc_yxLejvFMv(4VK~e0<CI5P=}Z;r`GA3_F<0_=G^MevE%95rgl8-"
    "TJ?bq;a`HSYj+;6<ewsabYD4*`7X*nG()Tbb>&PCQ0`Vtz(Va5_k00q)r$SBH^|gpHqUvB8(ylYTj$xNbBn%>(7TD4A-"
    "`ydUs)ImeptT`jbsFw947Jv3usLg?kdBX5e=revTResZ0EtVSdM+CLjsn(l)x!(q^c>B8fxWxV7D@HM5){WW0iVuVZ%2eF{q+iUz"
    "rCV)rDWsXyuw)%E3*Wd;a}Qw6iCmWS$yNRpwI3!w&<>|kz|&`Y`oMqyK09^J#m7LiDa#}_IaxIl;U7(Tjxv}pG{%t#{^dgS34TYC"
    "(ka`RQ2M)jRH8{3A5U`iOR1TNbWHTvMo$_icTij+Uf-"
    "DfmS6svjvk!ku#`PwyBOF(?n^rm;WC6x*@C-jszQqADwYW)C{=NP?^%>C9IY-Rjl`Ke)IOtZ{K`bN!YeZZb)N|qf4_SjoBY8MSG_"
    "blGzH(iDPRV333Oe=pGp?(xkbAtl`tT4j<$ic<t#pG>Cxu?wbN^!ZV$^z=`OL#hlM?jKmxz@0^v5+lGUXsQ<ZPRMf+>9#8BVG3^x"
    "waw<}|k|hX$V}0F>Qb`z{E>0Ln$3L@KIgV(Touk^<Q&9jf?F0+2BHz&b^lE(^(`?}OF6wvyz*CIr-"
    "P=M*3#_*o;E?ZZ^WgFcmOfa12y?o*v?SRp9CAam_Hy|dDmXz$XB_}BK1_6yhhvj$QM$IP9MZ+UvyCcnifhY3nNVmigBk(FX#^0^!"
    "Rrjc8zySODRON#`jYKUy9`_F%x_?FW^>e1m|O2emv9C~aHhs=LH-SF8@9clSzVoI_yVUlr&`wE{Ppc$-"
    "u(HwNB<XRpLbBW>=jH`tbL;5^u}~lhDJ)g&2dS^SRu&1l-^KjsT6et)^pTMob+kyEU%-tr&Xf9Txxu%dFzs<!p-"
    "x)#jNiD?ATc?xbRiXqmG=?<3%E@(hol(ajR+VqCQ$#6Jt6lR**X@Va&jxm7@Ta%qDTgcvh81x`+@I0a>QFBLz4!{&0fs;nNycpI+"
    "A8PM|fZk9Y`lV|fv-"
    "rL(y7JcrY*=S+S6A+4yTDJD}&tDEMfz8hTd9<!S{Z#oJV;tW+WzFS`}|Jx$(H^ld4DmJ_koS?$F*BYL#RZWJ-mnTq*Of5f~LKd@@"
    "yfhJ7j(}IoYAqYj&S&KApX$|%LS6M~*mcihJW#Pcw=Ea`T}G(q-p?WCRS!`-F}mQ~U||saY=vZlSUzLLmf!sK`L}Og-"
    "u?8)PcL76`||F)=hL2-_qP7eIXM40OkY{*{@w?l!Y2QnNeHMRM1N?`z`M_kTB5fww9}?i%L)(wv!al70MD)UF_T-"
    "dirgMG-w{B##m12kx2--FN0K@rfzSDvruM=+E0u5RkD0xxBc8=zF2(i{A4TfCv*}%V5(u{}*^R?JvZP~ZENP+EuNQ9nW}llH(Z|^"
    "AJW4gDf!V*Z7OM=CP%7$U+NX~6O9W2!>)8NEA5^GyHb3&D&`xaO>Q%a@`9lskbnw-~)bF@MMSSk}wKaCnnvX-"
    "~)O56%IGoHYob^f-Nf|E!i?(DwuK@{wLkg;LuYn-qtExO}|9O?Ai!Xa3*fHmf8(}#kS>ie{6s-"
    "pL8NIO8ajk>yo)B9$os_?GFGaQYJK!(PXR^Lf@e}ME71bRN0X@<7`gt)*kS7sGHm3?fy2cI>G(H~+VZzXh9&75boHZ^f-"
    "l@HFruosLnqEDid19$O&_Nxlti(Po6m<`FRAcMlvu}1}a;&SO*0DX*$l(wkH8$~BWdqaAmQa7Ef0+1G9o564+VXbcMW;<bB10Wlp"
    "vJB=!W<|n{zh|x=vH%Pt&%O2I>K{Hw#Tu)I59Pvdj8mle`pCW%w{oX=5m^L+YOaWjVrn1={8$$MBDReX9hS23F$Jv8?nrkK`%v9>"
    "4L%n^^VNGTSMCppC@jOktchFz;cFQsD`^{*ohCwNH<vQl1IkTo!HbhEwY-yz_^<OGj`<LaeV@tCzyBbUvGn+EEW&LUCWlv=m~XIm"
    "I_IvG1Qh|@tE5HqtUPHmoljwdOZ3x7b(S-JOaLpEkp+-"
    "7$%>UUM;eZqftkkE8^vBterK(lpOBY9`tZ&{pD#^e`!nODmRe9_pLOl5XiTGXHET=3t8DHvrO|7w$}8Q-"
    "ke{07k}vu6BBbsb(f@!T;kMcVi^)*lCIHeH6-"
    "?e!Fg_}pKUll{`7VB8{H&*(?w(Es15sXKa{2{z$DME?!({tn1qKQUZ;}UR??RL^mh$7^Ls&^Qbd}XT=njq9D2sEDzP$@oC%N6Bd`"
    "!$`G|6aVwcc4tDJ|8O?b;>IrCp@9!VL!*HU3m&d>3t8>em=x+!|;x6^Q~Wjb$k>>}vCc?^i@Oy-"
    "Q%u42BN;b=Xt@>$L1BG8MURJz4eR8`NZqzshT5hyF1TOghrJK?<7X8{|sv#Ocv_nOnqabywUa5czt%0N@wr&NR(mSWx5+PAYyIq0"
    "$gWU(>ssvu>oiB}}OUXO-U+!^3s$6iI{8ftn{hP8z=krB5NPUi5zN2(vu9lgw5!l^9_R=8B$oJTeN@rL_^-NZjH6udXXv$-"
    "`g0E#9^6z*3cNK!Vn&UQT;ccGP|jq*I!HEPvYkpCw;J{OJ6n|K6+o_?z=ghHvFz4cB$bc*L|E`L35%$g-"
    "4v1g98qW(0gee5n(b3)elrAO&vI`+(pVxg$DWL>LPx;ytIL8=bmces4V-"
    "+7)??u+J#)_iG(LR?Q`>|Ue#Qt@5<35m36%q6A=_da&!UWXH7sbfvfYQSCPezjT=lf<(&b>1F{#~gRh-eYiH36+_ZNB7a|$Ie*d3"
    "_iBk7vYziYP7Q!e#)Cqb4VbyH>}%5M4>QAdft*cvoMQ8ShGy-+tr5$LlStUW2!I=k&1_tmzPp*FVEvOWtFry-"
    "}5NM2*Aihk5UpZ0bYi~16r2>Y-Sk-nmHbqj9?BH&*;x6=;#^4tWRLx(ml1<i6J)3EwnLnBW?*V*9Ru<%8TV$v4iPAT!jh-"
    "()JZT&%7p|mAK2F<sy5OXbw0(ZS8kw`!J#2M$QfjT{;RJfXmUDoUanPfk7;>sZ4?+|IUi~x2qXzuj)lRC$fiMesg~5J^ZCN>?sf&"
    "0OJ?mDvWn7gRe6&7--3S@2J6Hnszwp^Vx=edinN;mrvP>-"
    "h4TKF};2B<GXL3KmPRTk1ucEeEag#yEk9{<;}Z~zx)2vho64`?VB&({P4r~Z=TP3d3pA&h+m#x_wAdnKm7R9kIzQ-"
    "=J~weefaL($LDiCzWeFr`G3Fr^x@ledj9a|4`08$@iRZY`|#aY|LU)J`{wuWK7H*AzxnX;@#`PIdVb-"
    "2^_wsE`oUlQ`Q^j!e($^e<@0x|SCj8pf9&sCf5~_La`DG}cOU)h+c(eI?|MG;r*|K}`kNoN)yJPtdESlZ%RYSl^^3FOb9=1I^V@y"
    "=^ybSnW9GMS-hce?gTF1$*L?HU`|n@A@weyKefjRE*T2@M{`UP>fB5+NvF8Kde0q8Lzb{|?@cqXR@BhzN&l~aGAKrZV&AX33zPRi"
    "kU*7-poKk=K^x^&UX8iW!hwr|2#om4U?Yq~P{H^cwtWs~@KR@pI{Xf0@`FV{$zJ8bf@Vl@7^zOT#-u>?7$MJsu{JsD1-"
    "G`rD*f;&BfB5uq{ATZe|MLD1KYah;c|||^F7p@v`^WFUd-"
    "J)p`P0j%A3r=F`SQ(&AHM$ecQ1eZ^zP%Y`yYS&^`GW9eJi|t{nekJo9p%7`t8f}mVWrjZJ95h_x{+QysdwE^V|Q^@5s;J?Pqg(p0"
    "s70!Cl#U@1L^tizs)-f1mG!=j5R)vX9$g-{+^K-=F*7-JgEvuge!Jv!GA-"
    "{!?zazx^romgE<GR^TsadzpRmuYYw(zy0Ck%a30@4~&<xZTY(AzJ2-"
    "AhxgCVeDnT~U%&g+zxmJq<DdS+KmGe({o~7j{QlGXmtXzY|MowB1#t4K|Mc#MpT7U$SO555{`<dmxt@P?ot9_2_+t8Z@BjF$4Y^?"
    "7z5o5U?>_zEt7ZBZS7qO53<O(|=|1)8xGLM0rVh8Fn6B$*S7o;aP|Hj?PRyS#&#JFN%kHH4Y`x<;J-"
    "2o`T;_w%|2{7}H|AG!ew`oRfBz|8etjV;BEuBhp7rJ3cVB&a=LWyNc)p!K&d=*R|6!i;a)0n0#$WqASWa)y*Y2)&pFTWK<a0~@|E"
    "|FEjyd5iaS$)-=PPiNFOT2ZP4@W;JTLV13t)d=ufX*Mr_%VB>-7peR}<f`|6%@x-"
    "|!bM$a80+VZpp4$LgP7kLh1~J@&u;^{@XA{?<d1"
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

class _ExploitTracker:
    """O(1)-per-round bookkeeping for the two exploitation rules."""

    def _reset_exploit(self) -> None:
        self._gap_upto = 1
        self._gap_n_c = 0
        self._gap_n_d = 0
        self._gap_d_c = 0
        self._gap_d_d = 0

    def _update_exploit(self, mine: list[int], theirs: list[int]) -> None:
        if self.first_d is None and theirs and theirs[-1] == D:
            self.first_d = len(theirs) - 1
        upto = self._gap_upto
        limit = len(theirs)
        while upto < limit:
            if mine[upto - 1] == C:
                self._gap_n_c += 1
                self._gap_d_c += theirs[upto] == D
            else:
                self._gap_n_d += 1
                self._gap_d_d += theirs[upto] == D
            upto += 1
        self._gap_upto = upto

    def _gap_counts(self):
        return (self._gap_n_c, self._gap_n_d,
                self._gap_d_c, self._gap_d_d)


class _AdversarialPolicy(_ExploitTracker):
    """Base policy + constrained late/non-responsive exploitation."""

    def __init__(self, base, gap=0.10, min_c=8, min_d=3,
                 late_day=50, repair=False):
        self.base = base
        self.gap = gap
        self.min_c = int(min_c)
        self.min_d = int(min_d)
        self.late_day = late_day
        self.repair = bool(repair)
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.punish = False
        self.first_d = None
        self._reset_exploit()

    def _observe(self, mine, theirs) -> None:
        observe = getattr(self.base, "observe", None)
        if observe is not None:
            observe(mine, theirs)

    def _start_punishment(self, mine, theirs) -> int:
        self.punish = True
        self._observe(mine, theirs)
        return D

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            self._observe(mine, theirs)
            return D
        if self.punish:
            if (self.repair and len(theirs) >= 2
                    and theirs[-1] == C and theirs[-2] == C):
                self.punish = False
                return self.base.act(t, mine, theirs)
            self._observe(mine, theirs)
            return D
        self._update_exploit(mine, theirs)
        if (self.late_day is not None and self.first_d is not None
                and self.first_d >= self.late_day
                and len(theirs) >= self.first_d + 2
                and theirs[self.first_d + 1] == D):
            return self._start_punishment(mine, theirs)
        if self.gap is not None:
            n_c, n_d, d_c, d_d = self._gap_counts()
            if n_c >= self.min_c and n_d >= self.min_d:
                p_c = d_c / n_c
                p_d = d_d / n_d
                if p_d - p_c <= self.gap:
                    return self._start_punishment(mine, theirs)
        return self.base.act(t, mine, theirs)


class AdvRL(Player):
    """Axelrod Player around the constrained adversarial RL policy."""

    name = "Adv-RL"
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
        _require_sample(mode, "Adv-RL")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = _AdversarialPolicy(_NeuralPolicy(_SPEC, self.seed), gap=0.2, min_c=12, min_d=2, late_day=40, repair=True)

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261024, mode: str = "sample",
          name: str | None = None, **_: object) -> AdvRL:
    return AdvRL(seed=seed, mode=mode, name=name)


__all__ = ["AdvRL", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((AdvRL(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{AdvRL.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")
