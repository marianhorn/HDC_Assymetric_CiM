%% i3thesis.cls
%% Copyright (C) 2009-2011 Christoph Sommer <christoph.sommer@informatik.uni-erlangen.de>
%%                         Falko Dressler
%%                    2014 Carsten Demel


%
% License (pick one or both):
% - Creative Commons Attribution-Share Alike 3.0 (or later)
% - GPL v2 (or later)
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{i3thesis}[2011/04/01 The I3 Thesis Class]  % YYYY/MM/DD


% -----------------------------------------------------------------------------
% Class options
% -----------------------------------------------------------------------------
\RequirePackage{ifthen}
\usepackage{advdate}

% [germanthesis] - Thesis is written in German
\newboolean{germanthesis}
\setboolean{germanthesis}{false}
\DeclareOption{germanthesis}{\setboolean{germanthesis}{true}}

% [plainunnumbered] - Don't print numbers on plain pages
\newboolean{plainunnumbered}
\setboolean{plainunnumbered}{false}
\DeclareOption{plainunnumbered}{\setboolean{plainunnumbered}{true}}

% [earlydraft] - Settings for quick draft printouts
\newboolean{earlydraft}
\setboolean{earlydraft}{false}
\DeclareOption{earlydraft}{\setboolean{earlydraft}{true}}

% [watermark] - Print current time/date at bottom of each page
\newboolean{watermark}
\setboolean{watermark}{false}
\DeclareOption{watermark}{\setboolean{watermark}{true}}

% [phdthesis] - switch to PhD thesis style
\newboolean{phdthesis}
\setboolean{phdthesis}{false}
\DeclareOption{phdthesis}{\setboolean{phdthesis}{true}}

% [twoside] - double sided
\newboolean{oneside}
\setboolean{oneside}{true}
\DeclareOption{twoside}{\setboolean{oneside}{false}}

\ProcessOptions \relax
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Basics
% -----------------------------------------------------------------------------
% This class is based on report
\ifthenelse{\boolean{oneside}}{
\LoadClass[a4paper,10pt,onecolumn,oneside,openright]{report}
\RequirePackage[a4paper,textwidth=345pt,textheight=598pt,bindingoffset=0cm,marginparwidth=3cm,hmarginratio=1:1,vmarginratio=1:1]{geometry}
}{
\LoadClass[a4paper,10pt,onecolumn,twoside,openright]{report}
\RequirePackage[a4paper,textwidth=345pt,textheight=598pt,bindingoffset=0cm,marginparwidth=3cm,hmarginratio=1:1,vmarginratio=1:1]{geometry}
}

% 150 percent line spacing
\RequirePackage{setspace}
\onehalfspacing

% Allow more (and larger) floats on text pages
\renewcommand{\topfraction}{0.9}  % use up to ..% of space on top
\renewcommand{\bottomfraction}{0.8}  % use up to ..% of space on bottom
\setcounter{topnumber}{4} % place up to .. on top
\setcounter{bottomnumber}{1} % place up to .. on bottom
\setcounter{totalnumber}{4} % place up to .. total
\renewcommand{\textfraction}{0.07} % allow down to ..% of text
\renewcommand{\floatpagefraction}{0.7} % fill at least ..% of float pages (must be less than \topfraction)

% Avoid widows and orphans (single line on bottom or top of page, respectively) at almost any cost
\clubpenalty = 10000
\widowpenalty = 10000
%\raggedbottom

% Default fonts
\newcommand{\mypageheadfont}{\normalfont}
\newcommand{\myheadingfont}{\normalfont}
%\RequirePackage[osf]{libertine}  % Linux Libertine
%\renewcommand{\mypageheadfont}{\normalfont\libertine}
%\renewcommand{\myheadingfont}{\normalfont\libertine}
\RequirePackage{lmodern}  % TT: Latin Modern
\RequirePackage{helvet}  % SF: Helvetica 
\RequirePackage[bitstream-charter,sfscaled=false]{mathdesign}  % RM: Bitstream Charter
% TODO: nicer math font?

% More encoding and typesetting fixes and tweaks
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\RequirePackage{fixltx2e}
\ifthenelse{\boolean{earlydraft}}{
	\newcommand{\mydisableprotrusion}{}
}{
	\RequirePackage[babel,protrusion=true,expansion=true,tracking=false,kerning=true,spacing=true]{microtype}
	\newcommand{\mydisableprotrusion}{\microtypesetup{protrusion=false}}
}
\RequirePackage{hyphenat}

% Graphics insertion and filetypes' order of precedence
\RequirePackage[pdftex]{graphicx}
\pdfminorversion=6
\DeclareGraphicsRule{.out.png}{png}{.out.png}{}
\ifthenelse{\boolean{earlydraft}}{
	\DeclareGraphicsExtensions{.thumb.jpg,.out.png,.pdf,.png,.jpg}
	\DeclareGraphicsRule{.thumb.jpg}{jpg}{.thumb.jpg}{}
}{
	\DeclareGraphicsExtensions{.out.png,.pdf,.png,.jpg}
}


% Typesetting URLs
\RequirePackage[lowtilde]{url}
\urlstyle{tt}

% For hyperref<->algorithm interaction
\RequirePackage{float}

% Cross-references are clickable
\RequirePackage[pdftex,bookmarks=true,bookmarksnumbered=true,colorlinks=false,pdfdisplaydoctitle=true,breaklinks=true]{hyperref}

% Captions
\RequirePackage[font=small, labelfont=bf, labelsep=endash, margin=1cm]{caption}

% Subfigures
\RequirePackage[margin=0cm]{subfig}

% The "SI" and "num" commands for typesetting units
%\RequirePackage[sepfour=true,tophrase=\dots,per=slash,obeyall]{siunitx}
\RequirePackage[range-phrase={\dots\,},per-mode=symbol,detect-all,load-configurations=binary,forbid-literal-units]{siunitx}
\DeclareSIUnit\byte{Byte}
\DeclareSIUnit\decibelm{dBm}
\DeclareSIUnit\mph{mph}

% Commands for beautiful tables
\RequirePackage{booktabs}

% Equation typesetting fixes and tweaks + proof/theorem environment
\RequirePackage{mathtools}
\RequirePackage{amsthm}
\newtheorem{thm}{Theorem}[section]
\everydisplay={\setlength{\abovedisplayshortskip}{-0.5em plus1ex minus1ex}\setlength{\abovedisplayskip}{-0.5em plus1ex minus1ex}}

% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Internationalization support
% -----------------------------------------------------------------------------

% Multi-language support using Babel
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[american,ngerman]{babel}
}{
	\RequirePackage[ngerman,american]{babel}
}

% % Don't indent paragraphs in german theses. Instead, leave blank space between paragraphs.
% \ifthenelse{\boolean{germanthesis}}{
% 	\RequirePackage{parskip}
% }{
% }

% Time and date functions
\RequirePackage{datetime}

% Clever References
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[ngerman]{cleveref}[2009/17/04]
	\def\cref{\Cref}
	\def\crefrange{\Crefrange}
	\crefname{lstlisting}{Listing}{Listings}
	\Crefname{lstlisting}{Listing}{Listings}
	\crefname{listing}{Listing}{Listings}
	\Crefname{listing}{Listing}{Listings}
	\crefname{algorithm}{Algorithmus}{Algorithmen}
	\Crefname{algorithm}{Algorithmus}{Algorithmen}
}{
	\RequirePackage{cleveref}[2009/17/04]
	\def\cref{\Cref}
	\def\crefrange{\Crefrange}
	\crefname{lstlisting}{listing}{listings}
	\Crefname{lstlisting}{Listing}{Listings}
	\crefname{listing}{listing}{listings}
	\Crefname{listing}{Listing}{Listings}
	\crefname{algorithm}{algorithm}{algorithms}
	\Crefname{algorithm}{Algorithm}{Algorithms}
}

% Some more language-dependent commands
\ifthenelse{\boolean{germanthesis}}{
    %\SetDate[16/01/2014]
	\newcommand{\thedate}{\today}
	\newcommand{\glossarytitlename}{Abkürzungsverzeichnis}
}{
    %\SetDate[16/01/2014]
	\newcommand{\thedate}{\selectlanguage{ngerman}\today\selectlanguage{american}}
	\newcommand{\glossarytitlename}{List of Acronyms}
}

% Language-independent timestamp
\newcommand{\timestamp}{\renewcommand{\dateseparator}{-}\yyyymmdddate\today\quad\settimeformat{hhmmsstime}\currenttime}

% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% All the simple stuff goes here
% -----------------------------------------------------------------------------

% The Acronym package
\RequirePackage[printonlyused]{acronym}

% Make automatically inserted "blank pages" really blank
\RequirePackage{emptypage}

% Fix spacing around "..."
\RequirePackage{ellipsis}

% TeX ist kein Zeichenprogramm
\RequirePackage{tikz}[2010/10/13]
\usetikzlibrary{}

% Ensure that no floats are typeset before they are referenced
\RequirePackage{flafter}

% Provide the \FloatBarrier command that flushes all pending floats
\RequirePackage{placeins}

% Nag about deprecated LaTeX packages and commands
\RequirePackage[l2tabu, orthodox]{nag}

% Define command that will return current filename
\RequirePackage{fink}

% Do not reset footnote counter on new chapter
\RequirePackage[]{chngcntr}
\counterwithout{footnote}{chapter}

% "Draft" watermark
\newcommand{\thewatermark}{\ifthenelse{\boolean{watermark}}{%
	\begin{tikzpicture}[remember picture, overlay]
		\node[anchor=south, outer sep=3.75cm] at (current page.south) {
			\color{gray}\finkfile\quad\timestamp
		};
	\end{tikzpicture}%
}{}}

% The "algorithm" environment
\RequirePackage{algorithmic}[2009/08/24]
\RequirePackage[plain,chapter]{algorithm}

\renewcommand\fs@plain{%
\def\@fs@cfont{\rmfamily}%
\let\@fs@capt\floatc@plain%
\def\@fs@pre{\hrule\relax\kern2pt}%
\def\@fs@post{}%
\def\@fs@mid{\kern2pt\hrule\relax\vspace\abovecaptionskip\relax}%
\let\@fs@iftopcapt\iffalse%
}
\floatstyle{plain}
\restylefloat{algorithm}
\ifthenelse{\boolean{germanthesis}}{
	\floatname{algorithm}{Algorithmus}
}{
}

% Chapter/section headings and page styles
%\RequirePackage[pagestyles,nobottomtitles*]{titlesec}
\RequirePackage[pagestyles]{titlesec}
\ifthenelse{\boolean{plainunnumbered}}{
\renewpagestyle{plain}{%
	\sethead{}{}{}
	\setfoot{}{}{\thewatermark}
}
}{
\renewpagestyle{plain}{%
	\sethead{}{}{}
	\setfoot{}{\mypageheadfont\thepage}{\thewatermark}
}
}

% Pagestyle "fancy"
\newpagestyle{fancy}{%
\sethead*{
	\bfseries
	\toptitlemarks
	\mypageheadfont\bfseries
	\ifthechapter
		{
			\ifthesection
			{\thesection~\sectiontitle}
			{\thechapter~\chaptertitle}
		}
		{\chaptertitle}
}
{}{\mypageheadfont\bfseries\thepage}
	\setfoot{}{}{\thewatermark}
	\headrule
}
\pagestyle{fancy}

% rename (sub)paragraph to (sub)subsubsubsection
\let\myparagraph\paragraph
\renewcommand{\paragraph}{\FIXME{in order to avoid confusion with hyperref, \\paragraph has been renamed to \\subsubsubsection}}
\let\subsubsubsection\myparagraph
\let\mysubparagraph\subparagraph
\renewcommand{\subparagraph}{\FIXME{in order to avoid confusion with hyperref, \\subparagraph has been renamed to \\subsubsubsubsection}}
\let\subsubsubsubsection\mysubparagraph

% heading styles
\titleformat{\chapter}[display]{\myheadingfont\huge\bfseries\titlerule\vspace{0.25cm}}{\LARGE\chaptertitlename~\thechapter}{1em}{}[\vspace{0.25cm}\titlerule]
\titleformat{\section}{\myheadingfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\myheadingfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\myheadingfont\normalsize\bfseries}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}{\myheadingfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titleformat{\subparagraph}{\myheadingfont\normalsize}{\thesubparagraph}{1em}{\labelitemi~}
\titlespacing*{\chapter}{0pt}{50pt}{40pt}
\titlespacing*{\section}{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex} 
\titlespacing*{\paragraph}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex} 
\titlespacing*{\subparagraph}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex} 
\ifthenelse{\boolean{earlydraft}}{
	\setcounter{tocdepth}{3}
	\setcounter{secnumdepth}{3}
}{
	\setcounter{tocdepth}{3}
	\setcounter{secnumdepth}{3}
}

% I3 Cover Page 
\RequirePackage{i3coverpage}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Commands for setting author, title, ... 
% -----------------------------------------------------------------------------
\renewcommand{\author}[1]{\gdef\theauthor{#1}}
\renewcommand{\title}[1]{\gdef\thetitle{#1}}
\newcommand{\germantitle}[1]{\gdef\thegermantitle{#1}}
\newcommand{\thesistype}[1]{\gdef\thethesistype{#1}}
\newcommand{\thesiscite}[1]{\gdef\thethesiscite{#1}}
\newcommand{\birthday}[1]{\gdef\thebirthday{#1}}
\newcommand{\birthplace}[1]{\gdef\thebirthplace{#1}}
\newcommand{\thesisstart}[1]{\gdef\thethesisstart{#1}}
\newcommand{\thesisend}[1]{\gdef\thethesisend{#1}}
\newcommand{\advisors}[1]{\gdef\theadvisors{#1}}
\newcommand{\dean}[1]{\gdef\thedean{#1}}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Listings
% -----------------------------------------------------------------------------
\RequirePackage[savemem]{listings}
\ifthenelse{\boolean{germanthesis}}{
	\renewcommand{\lstlistlistingname}{Listingverzeichnis}
}{
	\renewcommand{\lstlistlistingname}{List of Listings}
}
\lstset{
	language=C,
	frame=lines,
	framesep=5pt,
	captionpos=b,
	abovecaptionskip=1em,
	numbers=left,
	xleftmargin=15pt,
	framexleftmargin=15pt,
	numberstyle=\tiny,
	numbersep=5pt,
	stepnumber=1,
	fontadjust,
	tabsize=2,
	showtabs=false,
	showspaces=false,
	showstringspaces=false,
	breaklines=true,
	breakatwhitespace=true,
	keywordstyle=\color{black},
	identifierstyle=\bfseries\color{blue},
	commentstyle=\color{red},
	prebreak=\raisebox{0ex}[0ex][0ex]{\color{gray}\ensuremath{\searrow}},
	stringstyle=\bfseries\color{blue},
	basicstyle=\ttfamily\bfseries\color{black}\footnotesize
}
\lstdefinestyle{txt}{
	float,
	language=C
}
\lstdefinestyle{xml}{
	float,
	language=XML
}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% TOCs
% -----------------------------------------------------------------------------

% Include list-of-Xs in table of contents
\ifthenelse{\boolean{earlydraft}}{
	\RequirePackage[]{tocbibind}
	\renewcommand{\lstlistoflistings}{\begingroup\tocfile{\lstlistlistingname}{lol}\endgroup}
	\newcommand{\mymakeglossarytocentry}{\addcontentsline{toc}{chapter}{\glossarytitlename}}
	\newcommand{\mymaketodotocentry}{\todototoc}
}{
	\RequirePackage[nottoc,notlot,notlof]{tocbibind}
	\newcommand{\mymakeglossarytocentry}{}
	\newcommand{\mymaketodotocentry}{}
}

% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Visible TODO and FIXME markers
% -----------------------------------------------------------------------------
\ifthenelse{\boolean{germanthesis}}{
	\RequirePackage[colorinlistoftodos, textwidth=3.75cm, shadow, ngerman]{todonotes}
	\makeatletter
	\@todonotes@SetTodoListName{Offene Punkte}
	\makeatother
}{
	\RequirePackage[colorinlistoftodos, textwidth=3.75cm, shadow]{todonotes}
}
\newcounter{TODOCOUNT}
\newcommand{\TODO}[1]{\vspace{0.5em}\todo[inline, color=orange]{#1}\stepcounter{TODOCOUNT}}
\newcommand{\FIXME}[1]{\todo[size=\small, color=red]{#1}\stepcounter{TODOCOUNT}}
\AtEndDocument{
	\ifnum\value{TODOCOUNT}>0
		%\cleardoublepage
		\mymaketodotocentry\listoftodos
	\fi
}
% -----------------------------------------------------------------------------


% -----------------------------------------------------------------------------
% Title page
% -----------------------------------------------------------------------------
\renewcommand{\maketitle}{%

% Do some stuff we had to postpone until after e.g. \thetitle was set
\hypersetup{%
  pdftitle = {\thetitle},
  pdfsubject = {\thethesistype},
  pdfauthor = {\theauthor},
  pdfborder=0 0 0
}

\pagenumbering{alph}

% Output cover page
\makecoverpage{%
	%Department Informatik
	Lehrstuhl für Informatik 3%
}{%
	Rechnerarchitektur%
}{%
	\coverpagetextsmall{\nohyphens{\theauthor}}%
	\coverpagetextlarge{\nohyphens{\thetitle}}%
	\coverpagetextsmall{\thethesistype}%
	\coverpagetextsmall{\thethesisend}%
}{%
\begin{otherlanguage*}{american}%
	Please cite as:\\%
	\nohyphens{\theauthor, ``\thetitle,'' \thethesiscite, University of Erlangen, Dept. of Computer Science, September~\the\year.}%
\end{otherlanguage*}
}{%
	Friedrich-Alexander-Universität Erlangen-Nürnberg\\%
	Department Informatik\\%
	Rechnerarchitektur\\%
	\vspace{0.5em}%
	Martensstr. 3 $\cdot$ 91058 Erlangen $\cdot$ Germany%
}{%
	\href{http://www3.cs.fau.de/}{www3.cs.fau.de}%
}{%
	\includegraphics{i3/logo-tf_neu}%
}{%
	\includegraphics{i3/logo-fau_neu}%
}

% Title pages use single line spacing, no parindent and no parskips
\begin{singlespace}
\setlength\parindent{0pt} 
\setlength\parskip{0pt} 

%
% Now output the title page
%

\pagestyle{empty}
\pagenumbering{roman}

\ifthenelse{\boolean{phdthesis}}
{
\thispagestyle{empty}

\vspace*{1.5cm}

\begin{center}
\begin{singlespace}
\LARGE
\textbf{\thetitle}\\
\vspace*{.5\baselineskip}
---\\
\vspace*{.5\baselineskip}
\textbf{\thegermantitle}\\
\end{singlespace}

\vspace*{3cm}

\begin{doublespacing}
\large
Der Technischen Fakultät der\\
Universität Erlangen-Nürnberg\\
zur Erlangung des Grades\\
~\\
D\,O\,K\,T\,O\,R\,-\,I\,N\,G\,E\,N\,I\,E\,U\,R\\
~\\
vorgelegt von\\
~\\
\theauthor\\
~\\
Erlangen -- \the\year\\
\end{doublespacing}
\end{center}

\clearpage

\thispagestyle{empty}

\vspace*{14cm}

\begin{center}

\begin{onehalfspacing}
Als Dissertation genehmigt von\\

der Technischen Fakultät der\\

Universität Erlangen-Nürnberg\\[2\baselineskip]

\begin{tabular}{>{\raggedleft}p{5cm}p{7cm}}
Tag der Einreichung: & \textbf{\thethesisstart}\\
Tag der Promotion: & \textbf{\thethesisend}\\
Dekan: & \textbf{\thedean}\\
Berichterstatter: & \textbf{\theadvisors}\\
\end{tabular}
\end{onehalfspacing}

\end{center}

\cleardoublepage
}
% else: Master Thesis title page
{
\thispagestyle{empty}

\vspace*{0.8cm}

\begin{center}

{\LARGE \textbf{\thetitle}}\\[2\baselineskip]

{\large
Masterarbeit in Computational Engineering\\[2\baselineskip]

vorgelegt von\\[2\baselineskip]

\textbf{\theauthor}\\[1\baselineskip]

geb. am \thebirthday\\
in \thebirthplace\\[2\baselineskip]

angefertigt am\\[2\baselineskip]

\textbf{Lehrstuhl für Informatik 3\\[0.2\baselineskip]
Rechnerarchitektur\\[1.2\baselineskip]
Department Informatik\\[0.2\baselineskip]
Friedrich-Alexander-Universität Erlangen-Nürnberg\\[0.2\baselineskip]
}
}~\\[1\baselineskip]

\begin{tabular}{>{\raggedleft}p{5cm}p{6cm}}
Betreuender Hochschullehrer: & \textbf{Prof. Dr.-Ing. Dietmar Fey}\\

~ & ~\\

Beginn der Arbeit: & \textbf{\thethesisstart}\\
Abgabe der Arbeit: & \textbf{\thethesisend}\\
\end{tabular}

\end{center}

\clearpage
}
\end{singlespace}

%
% The delaration comes here
%

\ifthenelse{\boolean{phdthesis}}{}
{
\thispagestyle{empty}

\begin{otherlanguage*}{ngerman}
{\Large\noindent%
Erklärung \newline}

\noindent%
Ich versichere, dass ich die Arbeit ohne fremde Hilfe und ohne Benutzung anderer als der angegebenen Quellen angefertigt habe und dass die Arbeit in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen hat und von dieser als Teil einer Prüfungsleistung angenommen wurde.\\
Alle Ausführungen, die wörtlich oder sinngemäß übernommen wurden, sind als solche gekennzeichnet.
\end{otherlanguage*}


\vspace{2cm}


\begin{otherlanguage*}{american}
{\Large\noindent%
Declaration \newline}

\noindent%
I declare that the work is entirely my own and was produced with no assistance from third parties.\\
I certify that the work has not been submitted in the same or any similar form for assessment to any other examining body and all references, direct and indirect, are indicated as such and have been cited accordingly.
\end{otherlanguage*}


\vspace{3cm}


\noindent%
(\theauthor)\\
Erlangen, 01. September 2026

\cleardoublepage
}

\pagestyle{fancy}

}

% More encoding and typesetting fixes and tweaks
\usepackage{newunicodechar,textgreek}
\newunicodechar{µ}{\textmu}
\newunicodechar{α}{$\alphaup$}
\newunicodechar{β}{$\betaup$}
\newunicodechar{γ}{$\gammaup$}
\newunicodechar{δ}{$\deltaup$}
\newunicodechar{ε}{$\varepsilonup$} %\epsilonup your choice
\newunicodechar{ζ}{$\zetaup$}
\newunicodechar{η}{$\etaup$}
\newunicodechar{ϑ}{$\thetaup$}
\newunicodechar{ι}{$\iotaup$}
\newunicodechar{κ}{$\kappaup$}
\newunicodechar{λ}{$\lambdaup$}
\newunicodechar{μ}{$\muup$}
\newunicodechar{ν}{$\nuup$}
\newunicodechar{ξ}{$\xiup$}
\newunicodechar{ℴ}{o}
\newunicodechar{π}{$\piup$}
\newunicodechar{ν}{$\rhoup$}
\newunicodechar{σ}{$\sigmaup$}
\newunicodechar{τ}{$\tauup$}
\newunicodechar{φ}{$\varphiup$} %\phiup your choice
\newunicodechar{χ}{$\chiup$}
\newunicodechar{ψ}{$\psiup$}
\newunicodechar{ω}{$\omegaup$}
\newunicodechar{τ}{$\tauup$}
% -----------------------------------------------------------------------------
\usepackage{i3headings}
\usepackage{i3bib}

% Commands to store the content of the abstracts
\newcommand{\englishabstract}[1]{\gdef\theenglishabstract{#1}}
\newcommand{\germanabstract}[1]{\gdef\thegermanabstract{#1}}

% Environment to print the English abstract
\newenvironment{englishabstractenv}{
   \clearpage
   \vspace*{1in}
   \begin{center}
      \textbf{Abstract}
   \end{center}
   \vspace{1em}
   \noindent
   \theenglishabstract
   \vspace{1em}
   \clearpage
}{}

% Environment to print the German abstract
\newenvironment{germanabstractenv}{
   \clearpage
   \vspace*{1in}
   \begin{center}
      \textbf{Zusammenfassung}
   \end{center}
   \vspace{1em}
   \noindent
   \thegermanabstract
   \vspace{1em}
   \clearpage
}{}

