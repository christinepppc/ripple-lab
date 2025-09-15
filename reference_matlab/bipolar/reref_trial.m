function reref_trial(rootDir, badCh)
% Create <trialDir>_re-referenced with chan###_ref folders containing bipolar LFPs.
% Assumes per-channel files at: <trialDir>/chan###/lfp.mat with variables: lfp, fs
% Keeps ONLY the "left" electrode of each horizontal neighbor pair (i - j).
%
% Example:
%   rereref_trial('/data/session134/trial001', layoutGrid_26x10, [98 99]) 

rootDir = '/vol/brains/bd3/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001';
if nargin < 3, badCh = []; end
assert(isfolder(rootDir), 'Root folder not found: %s', rootDir);

% ---------------- layout -> neighbor pairs (within-bank, horizontal)
[pairs_h, pairs_v] = make_bipolar_pairs_from_grid(badCh);
pairs = pairs_h;    % <<< choose horizontal
% pairs = pairs_v;  % <<< or choose vertical neighbors instead

% ---------------- output folder
outDir = [rootDir '_re-referenced'];
if ~isfolder(outDir), mkdir(outDir); end

% ---- discover which channels exist in this trial (simple & strict)
avail    = false(221,1);          % 1..220 used
chanPath = strings(221,1);

nFound = 0;
for ch = 1:220
    chDir = fullfile(rootDir, sprintf('chan%03d', ch));
    if ~isfolder(chDir), continue; end

    % Prefer exact name lfp.mat
    f = fullfile(chDir, '*.mat');
    if ~isfile(f)
        % Optional fallback: accept exactly ONE lfp*.mat (e.g., lfp_bp.mat)
        cand = dir(fullfile(chDir, '*.mat'));
        if numel(cand) == 1
            f = fullfile(chDir, cand(1).name);
        else
            continue;  % nothing usable here
        end
    end

    avail(ch)    = true;
    chanPath(ch) = string(chDir);
    nFound       = nFound + 1;
end

fprintf('Discovery: found %d channels with lfp.mat under %s\n', nFound, rootDir);

% ---------------- process each pair present in this trial
pairs_used = [];
nDone = 0;
for p = 1:size(pairs,1)
    i = pairs(p,1); j = pairs(p,2);
    if ~avail(i) || ~avail(j), continue; end

    mi = dir(fullfile(char(chanPath(i)), '*.mat'));
    mj = dir(fullfile(char(chanPath(j)), '*.mat'));
    
    if numel(mi) ~= 1 || numel(mj) ~= 1
        warning('Expected exactly one .mat in chan%03d or chan%03d; found %d and %d. Skipping.', ...
                i, j, numel(mi), numel(mj));
        continue;
    end
    
    fi = fullfile(mi(1).folder, mi(1).name);
    fj = fullfile(mj(1).folder, mj(1).name);
    
    S_i = load(fi);   % now load takes a filename
    S_j = load(fj);


    if ~isfield(S_i,'lfp') || ~isfield(S_j,'lfp')
        warning('Missing lfp in %s or %s; skipping %d-%d', chanPath(i), chanPath(j), i, j);
        continue;
    end

    T = min(numel(S_i.lfp), numel(S_j.lfp));
    if T == 0, continue; end
    lfp_bp = S_i.lfp(1:T) - S_j.lfp(1:T);  % i minus j (left minus right)

    % save under chan###_ref (anchor = i)
    outChanDir = fullfile(outDir, sprintf('chan%03d_ref', i));
    if ~isfolder(outChanDir), mkdir(outChanDir); end

    fs = 1000; %#ok<NASGU>
    pair = [i j]; %#ok<NASGU>
    note = sprintf('bipolar neighbor (horizontal): ch%d - ch%d (within bank)', i, j); %#ok<NASGU>
    lfp_ref = lfp_bp(:); %#ok<NASGU>
    save(fullfile(outChanDir, 'lfp_ref.mat'), 'lfp_ref','fs','pair','note','-v7.3');

    pairs_used(end+1,:) = [i j]; %#ok<AGROW>
    nDone = nDone + 1;
end

% write manifest
if ~isempty(pairs_used)
    save(fullfile(outDir, 'pairs_used.mat'), 'pairs_used');
end

% build a live boolean mask of anchors that already have _ref
anchorDone = false(221,1);

% mark whatever was just produced by the horizontal loop
for k = 1:size(pairs_used,1)
    anchorDone(pairs_used(k,1)) = true;
end

% also mark what already exists on disk (in case of re-runs)
d = dir(fullfile(outDir, 'chan???_ref'));
for k = 1:numel(d)
    tok = regexp(d(k).name,'^chan(\d{3})_ref$','tokens','once');
    if ~isempty(tok)
        anchorDone(str2double(tok{1})) = true;
    end
end
fprintf('Saved %d re-referenced channels to %s\n', nDone, outDir);


fprintf('--- Starting vertical referencing ---\n');
nVert = 0;

for p = 1:size(pairs_v,1)
    i = pairs_v(p,1); j = pairs_v(p,2);

    % live guard: skip immediately if anchor already done
    if anchorDone(i)
        fprintf('SKIP %d-%d: anchor already done\n', i, j);
        continue;
    end
    if ~avail(i) || ~avail(j)
        fprintf('SKIP %d-%d: not available (avail=%d,%d)\n', i,j,avail(i),avail(j));
        continue;
    end

    mi = dir(fullfile(char(chanPath(i)), '*.mat'));
    mj = dir(fullfile(char(chanPath(j)), '*.mat'));
    if numel(mi)~=1 || numel(mj)~=1
        fprintf('SKIP %d-%d: wrong # .mat files (%d,%d)\n', i,j,numel(mi),numel(mj));
        continue;
    end

    fi = fullfile(mi(1).folder, mi(1).name);
    fj = fullfile(mj(1).folder, mj(1).name);
    S_i = load(fi); S_j = load(fj);

    if ~isfield(S_i,'lfp') || ~isfield(S_j,'lfp')
        fprintf('SKIP %d-%d: missing lfp var (%d,%d)\n', i,j,isfield(S_i,'lfp'),isfield(S_j,'lfp'));
        continue;
    end

    xi = double(S_i.lfp(:));
    xj = double(S_j.lfp(:));
    T  = min(numel(xi), numel(xj));
    if T==0
        fprintf('SKIP %d-%d: empty signal\n', i,j);
        continue;
    end
    lfp_ref = xi(1:T) - xj(1:T);

    % fs: prefer i, then j, else NaN
    if isfield(S_i,'fs'), fs = S_i.fs;
    elseif isfield(S_j,'fs'), fs = S_j.fs;
    else, fs = NaN; end

    pair = [i j];
    note = sprintf('bipolar neighbor (vertical): ch%d - ch%d (within bank)', i, j);

    outChanDir = fullfile(outDir, sprintf('chan%03d_ref', i));
    if ~isfolder(outChanDir), mkdir(outChanDir); end
    save(fullfile(outChanDir,'lfp_ref.mat'),'lfp_ref','fs','pair','note','-v7.3');

    fprintf('SAVED vertical pair %d-%d\n', i, j);

    % *** THIS IS THE CRUCIAL LINE ***
    anchorDone(i) = true;     % mark immediately so later pairs for this anchor are skipped

    nVert = nVert + 1;
end

fprintf('Vertical referencing saved %d new channels.\n', nVert);