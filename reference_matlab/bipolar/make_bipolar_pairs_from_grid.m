function [pairs_h, pairs_v] = make_bipolar_pairs_from_grid(badCh)
% Build horizontal and vertical bipolar pairs from a channel-ID grid.
% - Only pairs within the SAME bank (headstage)
% - Skips blanks (0) and user-specified bad channels
% - 1-based channel IDs (your data)
%
% Returns:
%   pairs_h: [K_h x 2] [i j] with j = right neighbor (i.e., y = x_i - x_j)
%   pairs_v: [K_v x 2] [i j] with j = below neighbor
%
% Sign convention here: y = x(i) - x(j)
% Horizontal: left minus right; Vertical: top minus bottom.

if nargin < 2, badCh = []; end

layoutGrid_26x10 = [ ... 
    2 1 4 3 98 0 0 0 0 0;
    6 5 8 7 97 0 0 0 0 0;
    102 101 108 10 100 99 0 0 0 0;
    104 103 110 9 11 109 0 0 0 0;
    106 105 107 12 14 13 22 0 0 0;
    16 15 18 17 20 19 21 0 0 0;
    24 23 27 114 30 29 116 115 0 0;
    26 25 111 113 32 31 118 117 0 0;
    28 34 33 120 119 124 126 125 128 0;
    112 36 35 38 122 123 40 39 127 0;
    130 129 134 42 37 121 44 43 46 45;
    132 131 133 41 56 55 136 135 59 62;
    48 47 54 53 58 57 138 137 149 61;
    50 49 140 142 144 60 148 150 152 64;
    52 51 139 141 143 146 145 147 151 63;
    162 66 65 161 164 163 154 156 158 160;
    166 68 67 165 168 167 153 155 157 159;
    170 169 172 171 174 173 175 178 70 177;
    180 179 182 181 72 74 176 75 77 69;
    184 183 186 185 71 73 76 78 80 191;
    188 187 190 189 84 83 86 85 88 192;
    194 79 193 196 195 199 89 94 96 87;
    198 82 81 197 200 90 92 93 95 0;
    202 201 204 203 206 205 91 208 0 0;
    207 210 209 212 211 214 213 0 0 0;
    216 215 218 217 220 219 0 0 0 0 ];

G = layoutGrid_26x10;        % size 26 x 10
[R, C] = size(G);

% ---- bank lookup (Banks 1..7, inclusive)
bankOf = @(ch) ...
    (ch>=1   && ch<=32 ) * 1 + ...
    (ch>=33  && ch<=64 ) * 2 + ...
    (ch>=65  && ch<=96 ) * 3 + ...
    (ch>=97  && ch<=128) * 4 + ...
    (ch>=129 && ch<=160) * 5 + ...
    (ch>=161 && ch<=192) * 6 + ...
    (ch>=193 && ch<=220) * 7;

isBad = false(221,1);          % quick mask up to 220
isBad(badCh(:)) = true;

pairs_h = [];  % horizontal: (r,c) - (r,c+1)
for r = 1:R
    for c = 1:(C-1)
        i = G(r,c);
        j = G(r,c+1);
        if i==0 || j==0, continue; end
        if isBad(i) || isBad(j), continue; end
        if bankOf(i) ~= bankOf(j), continue; end   % stay within bank
        pairs_h(end+1,:) = [i j]; %#ok<AGROW>
    end
end

pairs_v = [];  % vertical: (r,c) - (r+1,c)
for r = 1:(R-1)
    for c = 1:C
        i = G(r,c);
        j = G(r+1,c);
        if i==0 || j==0, continue; end
        if isBad(i) || isBad(j), continue; end
        if bankOf(i) ~= bankOf(j), continue; end
        pairs_v(end+1,:) = [i j]; %#ok<AGROW>
    end
end
end