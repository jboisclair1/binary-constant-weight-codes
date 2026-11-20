using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace AugmentingTradeSearch;

internal static class Program
{
    private const string Version = "ATS v1/v2 standalone";

    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        try
        {
            Options options = Options.Parse(args);
            if (options.ShowHelp)
            {
                Options.PrintHelp();
                return 0;
            }

            if (string.IsNullOrWhiteSpace(options.InputPath))
                throw new UserError("--input is required.");
            Code code = CodeFile.Read(options.InputPath, options.ZeroBasedInput, options.N);
            int n = options.N ?? code.InferredLength;
            int w = options.W ?? code.InferredWeight;
            int d = options.D ?? throw new UserError("--d is required.");
            ValidateParameters(n, d, w);
            VerificationResult initial = Verifier.Verify(code.Words, n, d, w);
            if (!initial.Valid)
            {
                throw new UserError($"Input code is invalid: {initial.Message}");
            }

            Console.WriteLine($"{Version}: loaded {code.Words.Count:N0} valid ({n},{d},{w}) words.");
            if (options.Command == Command.Verify)
            {
                Console.WriteLine($"Verified minimum distance ≥ {d}; maximum pair intersection is {initial.MaxIntersection}.");
                return 0;
            }

            int reservoir = 0;
            if (options.Command != Command.SearchV2)
            {
                reservoir = options.Reservoir is null
                    ? ChooseSparsestReservoir(code.Words, n)
                    : ParseCoordinate(options.Reservoir, n);
                int usage = code.Words.Count(word => (word & Bit(reservoir)) != 0);
                Console.WriteLine($"Reservoir: {reservoir + 1} ({usage:N0} existing words use it).");
            }
            else
            {
                Console.WriteLine("ATS v2: independent replacement coordinates enabled.");
            }

            ISearch search = options.Command == Command.SearchV2
                ? new AtsV2Search(code.Words, n, d, w, options)
                : new AtsSearch(code.Words, n, d, w, reservoir, options);
            IReadOnlyList<SearchResult> results = search.Run();
            if (results.Count == 0)
            {
                Console.WriteLine($"No {(options.Command == Command.SearchV2 ? "ATS v2" : "ATS v1")} augmentation found within the configured limits.");
                Console.WriteLine(
                    $"Checked {search.CandidatesEnumerated:N0} candidates; " +
                    $"{search.CandidatesAdmissible:N0} passed the intersection filter; " +
                    $"elapsed {search.Elapsed.TotalSeconds:F3}s.");
                return 2;
            }

            for (int i = 0; i < results.Count; i++)
            {
                SearchResult result = results[i];
                string requestedOutput = options.OutputPath == Options.AutoOutput
                    ? Path.Combine("Discoveries", $"A_{n}_{d}_{w}_lb{result.Code.Count}_v2.txt")
                    : options.OutputPath;
                string output = OutputPath(requestedOutput, i, results.Count);
                CodeFile.Write(output, result.Code, n);
                string certificate = Path.ChangeExtension(output, ".trade.txt");
                CertificateFile.Write(certificate, result, n, d, w, options.Command);
                Console.WriteLine(
                    $"Result {i + 1}: {code.Words.Count}→{result.Code.Count}, " +
                    $"{result.Blockers.Count}→{result.Blockers.Count + 1} trade, " +
                    $"E={FormatWord(result.ActiveWord, n)}");
                Console.WriteLine($"  Wrote {output}");
                Console.WriteLine($"  Wrote {certificate}");
            }

            Console.WriteLine(
                $"Checked {search.CandidatesEnumerated:N0} candidates; " +
                $"{search.CandidatesAdmissible:N0} passed the intersection filter; " +
                $"elapsed {search.Elapsed.TotalSeconds:F3}s.");
            return 0;
        }
        catch (UserError error)
        {
            Console.Error.WriteLine($"error: {error.Message}");
            return 1;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"fatal: {error}");
            return 1;
        }
    }

    private static void ValidateParameters(int n, int d, int w)
    {
        if (n is < 1 or > 64)
            throw new UserError("This optimized build supports 1 ≤ n ≤ 64.");
        if (w < 1 || w > n)
            throw new UserError("Weight must satisfy 1 ≤ w ≤ n.");
        if (d < 2 || d > 2 * w || (d & 1) != 0)
            throw new UserError("Distance must be an even integer satisfying 2 ≤ d ≤ 2w.");
    }

    private static int ChooseSparsestReservoir(IReadOnlyList<ulong> words, int n)
    {
        int best = 0;
        int bestCount = int.MaxValue;
        for (int coordinate = 0; coordinate < n; coordinate++)
        {
            ulong bit = Bit(coordinate);
            int count = words.Count(word => (word & bit) != 0);
            if (count < bestCount)
            {
                best = coordinate;
                bestCount = count;
            }
        }
        return best;
    }

    private static int ParseCoordinate(string text, int n)
    {
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out int value) || value < 1 || value > n)
            throw new UserError($"Reservoir must be a 1-based coordinate from 1 through {n}.");
        return value - 1;
    }

    internal static ulong Bit(int coordinate) => 1UL << coordinate;

    internal static string FormatWord(ulong word, int n)
    {
        var coordinates = new List<int>();
        for (int i = 0; i < n; i++)
            if ((word & Bit(i)) != 0)
                coordinates.Add(i + 1);
        return "{" + string.Join(',', coordinates) + "}";
    }

    private static string OutputPath(string requested, int index, int count)
    {
        if (count == 1)
            return Path.GetFullPath(requested);
        string full = Path.GetFullPath(requested);
        string directory = Path.GetDirectoryName(full)!;
        string stem = Path.GetFileNameWithoutExtension(full);
        string extension = Path.GetExtension(full);
        return Path.Combine(directory, $"{stem}.{index + 1}{extension}");
    }
}

internal enum Command { Search, SearchV2, Verify }

internal sealed record Options(
    Command Command,
    string? InputPath,
    string OutputPath,
    int? N,
    int? D,
    int? W,
    string? Reservoir,
    int MaxBlockers,
    long MaxCandidates,
    int MaxResults,
    int SolutionsPerCandidate,
    bool ZeroBasedInput,
    bool ShowHelp)
{
    public const string AutoOutput = "<auto-discovery-output>";

    public static Options Parse(string[] args)
    {
        if (args.Length == 0)
            return Defaults(showHelp: true);

        int index = 0;
        Command command = Command.Search;
        if (args[0].Equals("verify", StringComparison.OrdinalIgnoreCase))
        {
            command = Command.Verify;
            index++;
        }
        else if (args[0].Equals("search", StringComparison.OrdinalIgnoreCase))
        {
            index++;
        }
        else if (args[0].Equals("search-v2", StringComparison.OrdinalIgnoreCase) ||
                 args[0].Equals("v2", StringComparison.OrdinalIgnoreCase))
        {
            command = Command.SearchV2;
            index++;
        }

        Options value = Defaults(command: command);
        while (index < args.Length)
        {
            string arg = args[index++];
            string Next()
            {
                if (index >= args.Length)
                    throw new UserError($"Missing value after {arg}.");
                return args[index++];
            }

            value = arg switch
            {
                "-h" or "--help" => value with { ShowHelp = true },
                "-i" or "--input" => value with { InputPath = Next() },
                "-o" or "--output" => value with { OutputPath = Next() },
                "--n" => value with { N = ParseInt(Next(), arg, 1) },
                "--d" => value with { D = ParseInt(Next(), arg, 2) },
                "--w" => value with { W = ParseInt(Next(), arg, 1) },
                "-r" or "--reservoir" => value with { Reservoir = Next() },
                "--max-blockers" => value with { MaxBlockers = ParseInt(Next(), arg, 0) },
                "--max-candidates" => value with { MaxCandidates = ParseLong(Next(), arg, 0) },
                "--max-results" => value with { MaxResults = ParseInt(Next(), arg, 1) },
                "--solutions-per-candidate" => value with { SolutionsPerCandidate = ParseInt(Next(), arg, 1) },
                "--zero-based" => value with { ZeroBasedInput = true },
                _ => throw new UserError($"Unknown argument: {arg}")
            };
        }

        return value;
    }

    private static Options Defaults(Command command = Command.Search, bool showHelp = false) =>
        new(command, null, AutoOutput, null, null, null, null, 24, 0, 1, 1, false, showHelp);

    private static int ParseInt(string text, string option, int minimum)
    {
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out int value) || value < minimum)
            throw new UserError($"{option} requires an integer ≥ {minimum}.");
        return value;
    }

    private static long ParseLong(string text, string option, long minimum)
    {
        if (!long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out long value) || value < minimum)
            throw new UserError($"{option} requires an integer ≥ {minimum}.");
        return value;
    }

    public static void PrintHelp() => Console.WriteLine(
        """
        ATS v1/v2 — augmenting-trade search for binary constant-weight codes

        Usage:
          ats search --input CODE.txt --d D [--n N] [--w W] [options]
          ats search-v2 --input CODE.txt --d D [--n N] [--w W] [options]
          ats verify --input CODE.txt --d D [--n N] [--w W]

        Input formats (one codeword per line; # starts a comment):
          {1,2,3,7,9}       1-based coordinate set
          1 2 3 7 9         delimited coordinates
          111000101         binary incidence vector

        Search options:
          -r, --reservoir N            v1 reservoir coordinate (default: sparsest)
          -o, --output PATH            result file
          --max-blockers N             skip candidates with >N blockers (default: 24)
          --max-candidates N           stop after N candidates; 0 is unlimited
          --max-results N              number of improved codes to write (default: 1)
          --solutions-per-candidate N  CSP branches retained per active word (default: 1)
          --zero-based                 parse coordinate-list inputs as zero-based

        ATS v1 uses one shared reservoir coordinate. ATS v2 permits independent
        replacement coordinates. The bitmask engine supports n ≤ 64. Output is
        always 1-based set notation.
        """);
}

internal interface ISearch
{
    long CandidatesEnumerated { get; }
    long CandidatesAdmissible { get; }
    TimeSpan Elapsed { get; }
    IReadOnlyList<SearchResult> Run();
}

internal sealed class AtsSearch : ISearch
{
    private readonly IReadOnlyList<ulong> _code;
    private readonly HashSet<ulong> _codeSet;
    private readonly int _n;
    private readonly int _w;
    private readonly int _lambda;
    private readonly int _reservoir;
    private readonly ulong _reservoirBit;
    private readonly Options _options;
    private readonly Stopwatch _clock = new();

    public long CandidatesEnumerated { get; private set; }
    public long CandidatesAdmissible { get; private set; }
    public TimeSpan Elapsed => _clock.Elapsed;

    public AtsSearch(IReadOnlyList<ulong> code, int n, int d, int w, int reservoir, Options options)
    {
        _code = code;
        _codeSet = code.ToHashSet();
        _n = n;
        _w = w;
        _lambda = w - d / 2;
        _reservoir = reservoir;
        _reservoirBit = Program.Bit(reservoir);
        _options = options;
    }

    public IReadOnlyList<SearchResult> Run()
    {
        _clock.Start();
        var results = new List<SearchResult>();
        var seenCodes = new HashSet<string>(StringComparer.Ordinal);

        foreach (ulong active in EnumerateWeightWords(_n, _w, _reservoir))
        {
            CandidatesEnumerated++;
            if (_options.MaxCandidates != 0 && CandidatesEnumerated > _options.MaxCandidates)
                break;
            if (_codeSet.Contains(active))
                continue;

            Candidate? candidate = AnalyzeCandidate(active);
            if (candidate is null)
                continue;
            CandidatesAdmissible++;

            foreach (ulong[] deletionBits in Solve(candidate, _options.SolutionsPerCandidate))
            {
                SearchResult? result = BuildAndVerify(candidate, deletionBits);
                if (result is null)
                    continue;
                string signature = string.Join(',', result.Code.Order());
                if (!seenCodes.Add(signature))
                    continue;
                results.Add(result);
                if (results.Count >= _options.MaxResults)
                {
                    _clock.Stop();
                    return results;
                }
            }
        }

        _clock.Stop();
        return results;
    }

    private Candidate? AnalyzeCandidate(ulong active)
    {
        var blockers = new List<ulong>();
        foreach (ulong word in _code)
        {
            int intersection = BitOperations.PopCount(word & active);
            if (intersection > _lambda + 1)
                return null;
            if (intersection == _lambda + 1)
            {
                if ((word & _reservoirBit) != 0)
                    return null; // A one-point lift would not preserve weight.
                blockers.Add(word);
                if (blockers.Count > _options.MaxBlockers)
                    return null;
            }
        }

        ulong[] blockerArray = blockers.ToArray();
        var blockerSet = blockerArray.ToHashSet();
        ulong[] domains = new ulong[blockerArray.Length];
        for (int i = 0; i < blockerArray.Length; i++)
        {
            ulong domain = blockerArray[i] & active;
            foreach (ulong survivor in _code)
            {
                if (blockerSet.Contains(survivor) || (survivor & _reservoirBit) == 0)
                    continue;
                int baseIntersection = BitOperations.PopCount(blockerArray[i] & (survivor & ~_reservoirBit));
                if (baseIntersection + 1 > _lambda)
                    domain &= survivor;
            }
            if (domain == 0)
                return null;
            domains[i] = domain;
        }

        var constraints = new List<PairConstraint>();
        for (int i = 0; i < blockerArray.Length; i++)
        {
            for (int j = i + 1; j < blockerArray.Length; j++)
            {
                ulong intersection = blockerArray[i] & blockerArray[j];
                if (BitOperations.PopCount(intersection) == _lambda)
                    constraints.Add(new PairConstraint(i, j, intersection));
            }
        }
        return new Candidate(active, blockerArray, domains, constraints.ToArray());
    }

    private static IEnumerable<ulong[]> Solve(Candidate candidate, int limit)
    {
        if (candidate.Blockers.Length == 0)
        {
            yield return Array.Empty<ulong>();
            yield break;
        }

        ulong[] assigned = new ulong[candidate.Blockers.Length];
        int yielded = 0;
        foreach (ulong[] solution in Search())
        {
            yield return solution;
            if (++yielded >= limit)
                yield break;
        }

        IEnumerable<ulong[]> Search()
        {
            int variable = -1;
            ulong available = 0;
            int bestSize = int.MaxValue;
            for (int i = 0; i < assigned.Length; i++)
            {
                if (assigned[i] != 0)
                    continue;
                ulong filtered = FilterDomain(i, candidate.Domains[i]);
                int size = BitOperations.PopCount(filtered);
                if (size == 0)
                    yield break;
                if (size < bestSize)
                {
                    variable = i;
                    available = filtered;
                    bestSize = size;
                }
            }

            if (variable < 0)
            {
                yield return (ulong[])assigned.Clone();
                yield break;
            }

            while (available != 0)
            {
                ulong choice = available & (0UL - available);
                available ^= choice;
                assigned[variable] = choice;
                if (AllAssignedConstraintsHold(variable))
                    foreach (ulong[] solution in Search())
                        yield return solution;
                assigned[variable] = 0;
            }
        }

        ulong FilterDomain(int variable, ulong domain)
        {
            foreach (PairConstraint constraint in candidate.Constraints)
            {
                int other;
                if (constraint.Left == variable) other = constraint.Right;
                else if (constraint.Right == variable) other = constraint.Left;
                else continue;
                if (assigned[other] != 0 && (assigned[other] & constraint.Intersection) == 0)
                    domain &= constraint.Intersection;
            }
            return domain;
        }

        bool AllAssignedConstraintsHold(int variable)
        {
            foreach (PairConstraint constraint in candidate.Constraints)
            {
                if (constraint.Left != variable && constraint.Right != variable)
                    continue;
                ulong left = assigned[constraint.Left];
                ulong right = assigned[constraint.Right];
                if (left != 0 && right != 0 &&
                    (left & constraint.Intersection) == 0 &&
                    (right & constraint.Intersection) == 0)
                    return false;
            }
            return true;
        }
    }

    private SearchResult? BuildAndVerify(Candidate candidate, ulong[] deletionBits)
    {
        var blockerSet = candidate.Blockers.ToHashSet();
        var result = new List<ulong>(_code.Count + 1);
        foreach (ulong word in _code)
            if (!blockerSet.Contains(word))
                result.Add(word);
        for (int i = 0; i < candidate.Blockers.Length; i++)
            result.Add((candidate.Blockers[i] & ~deletionBits[i]) | _reservoirBit);
        result.Add(candidate.Active);

        VerificationResult verification = Verifier.Verify(result, _n, 2 * (_w - _lambda), _w);
        return verification.Valid
            ? new SearchResult(result, candidate.Active, candidate.Blockers, deletionBits, deletionBits.Select(_ => _reservoirBit).ToArray())
            : null;
    }

    private static IEnumerable<ulong> EnumerateWeightWords(int n, int w, int excludedCoordinate)
    {
        int availableCount = n - 1;
        if (w > availableCount)
            yield break;
        ulong combination = (1UL << w) - 1;
        ulong limit = availableCount == 64 ? 0 : 1UL << availableCount;
        while (combination < limit)
        {
            ulong low = combination & ((1UL << excludedCoordinate) - 1);
            ulong high = combination & ~((1UL << excludedCoordinate) - 1);
            yield return low | (high << 1);

            ulong smallest = combination & (0UL - combination);
            ulong ripple = combination + smallest;
            if (ripple == 0)
                yield break;
            combination = (((ripple ^ combination) >> 2) / smallest) | ripple;
        }
    }
}

internal sealed class AtsV2Search : ISearch
{
    private readonly IReadOnlyList<ulong> _code;
    private readonly HashSet<ulong> _codeSet;
    private readonly int _n;
    private readonly int _d;
    private readonly int _w;
    private readonly int _lambda;
    private readonly ulong _universe;
    private readonly Options _options;
    private readonly Dictionary<ulong, int> _subsetOwner;
    private readonly int[] _seenStamp;
    private readonly ulong[] _blockerScratch;
    private readonly int _blockerSubsetSize;
    private int _currentStamp;
    private readonly Stopwatch _clock = new();

    public long CandidatesEnumerated { get; private set; }
    public long CandidatesAdmissible { get; private set; }
    public TimeSpan Elapsed => _clock.Elapsed;

    public AtsV2Search(IReadOnlyList<ulong> code, int n, int d, int w, Options options)
    {
        _code = code;
        _codeSet = code.ToHashSet();
        _n = n;
        _d = d;
        _w = w;
        _lambda = w - d / 2;
        _universe = n == 64 ? ulong.MaxValue : (1UL << n) - 1;
        _options = options;
        _blockerSubsetSize = _lambda + 1;
        _subsetOwner = new Dictionary<ulong, int>(code.Count * Math.Max(1, Binomial(w, _blockerSubsetSize)));
        for (int wordIndex = 0; wordIndex < code.Count; wordIndex++)
        {
            ulong word = code[wordIndex];
            for (ulong subset = word; subset != 0; subset = (subset - 1) & word)
            {
                if (BitOperations.PopCount(subset) != _blockerSubsetSize)
                    continue;
                if (!_subsetOwner.TryAdd(subset, wordIndex))
                    throw new UserError("Input violates the intersection bound required by the subset index.");
            }
        }
        _seenStamp = new int[code.Count];
        _blockerScratch = new ulong[code.Count];
    }

    public IReadOnlyList<SearchResult> Run()
    {
        _clock.Start();
        var results = new List<SearchResult>();
        var seenCodes = new HashSet<string>(StringComparer.Ordinal);
        foreach (ulong active in EnumerateWeightWords(_n, _w))
        {
            CandidatesEnumerated++;
            if (_options.MaxCandidates != 0 && CandidatesEnumerated > _options.MaxCandidates)
                break;
            if (_codeSet.Contains(active))
                continue;
            V2Candidate? candidate = Analyze(active);
            if (candidate is null)
                continue;
            CandidatesAdmissible++;

            foreach (RepairOption[] assignment in Solve(candidate, _options.SolutionsPerCandidate))
            {
                SearchResult? result = BuildAndVerify(candidate, assignment);
                if (result is null)
                    continue;
                string signature = string.Join(',', result.Code.Order());
                if (!seenCodes.Add(signature))
                    continue;
                results.Add(result);
                if (results.Count >= _options.MaxResults)
                {
                    _clock.Stop();
                    return results;
                }
            }
        }
        _clock.Stop();
        return results;
    }

    private V2Candidate? Analyze(ulong active)
    {
        if (++_currentStamp == int.MaxValue)
        {
            Array.Clear(_seenStamp);
            _currentStamp = 1;
        }
        int blockerCount = 0;
        for (ulong subset = active; subset != 0; subset = (subset - 1) & active)
        {
            if (BitOperations.PopCount(subset) != _blockerSubsetSize || !_subsetOwner.TryGetValue(subset, out int owner))
                continue;
            if (_seenStamp[owner] == _currentStamp)
                return null; // The same word owns multiple subsets, so its intersection is too large.
            _seenStamp[owner] = _currentStamp;
            _blockerScratch[blockerCount++] = _code[owner];
            if (blockerCount > _options.MaxBlockers)
                return null;
        }

        ulong[] blockerArray = new ulong[blockerCount];
        Array.Copy(_blockerScratch, blockerArray, blockerCount);
        var survivors = new ulong[_code.Count - blockerCount];
        int survivorCount = 0;
        for (int wordIndex = 0; wordIndex < _code.Count; wordIndex++)
            if (_seenStamp[wordIndex] != _currentStamp)
                survivors[survivorCount++] = _code[wordIndex];
        var domains = new RepairOption[blockerArray.Length][];
        for (int i = 0; i < blockerArray.Length; i++)
        {
            ulong blocker = blockerArray[i];
            ulong deletions = blocker & active;
            ulong additions = _universe & ~blocker & ~active;
            var options = new List<RepairOption>();
            while (deletions != 0)
            {
                ulong deletion = LowestBit(deletions);
                deletions ^= deletion;
                ulong remainingAdditions = additions;
                while (remainingAdditions != 0)
                {
                    ulong addition = LowestBit(remainingAdditions);
                    remainingAdditions ^= addition;
                    ulong lifted = (blocker & ~deletion) | addition;
                    bool compatible = true;
                    foreach (ulong survivor in survivors)
                    {
                        if (BitOperations.PopCount(lifted & survivor) > _lambda)
                        {
                            compatible = false;
                            break;
                        }
                    }
                    if (compatible)
                        options.Add(new RepairOption(deletion, addition, lifted));
                }
            }
            domains[i] = options.ToArray();
            if (domains[i].Length == 0)
                return null;
        }
        return new V2Candidate(active, blockerArray, domains);
    }

    private IEnumerable<RepairOption[]> Solve(V2Candidate candidate, int limit)
    {
        if (candidate.Blockers.Length == 0)
        {
            yield return Array.Empty<RepairOption>();
            yield break;
        }
        var assigned = new RepairOption?[candidate.Blockers.Length];
        int yielded = 0;
        foreach (RepairOption[] solution in Search())
        {
            yield return solution;
            if (++yielded >= limit)
                yield break;
        }

        IEnumerable<RepairOption[]> Search()
        {
            int variable = -1;
            List<RepairOption>? available = null;
            for (int i = 0; i < assigned.Length; i++)
            {
                if (assigned[i] is not null)
                    continue;
                var filtered = candidate.Domains[i]
                    .Where(option => CompatibleWithAssigned(option, assigned))
                    .ToList();
                if (filtered.Count == 0)
                    yield break;
                if (available is null || filtered.Count < available.Count)
                {
                    variable = i;
                    available = filtered;
                }
            }
            if (variable < 0)
            {
                yield return assigned.Select(option => option!).ToArray();
                yield break;
            }
            foreach (RepairOption option in available!)
            {
                assigned[variable] = option;
                foreach (RepairOption[] solution in Search())
                    yield return solution;
                assigned[variable] = null;
            }
        }
    }

    private bool CompatibleWithAssigned(RepairOption option, RepairOption?[] assigned)
    {
        foreach (RepairOption? other in assigned)
            if (other is not null && BitOperations.PopCount(option.Lifted & other.Lifted) > _lambda)
                return false;
        return true;
    }

    private SearchResult? BuildAndVerify(V2Candidate candidate, RepairOption[] assignment)
    {
        var blockerSet = candidate.Blockers.ToHashSet();
        var result = new List<ulong>(_code.Count + 1);
        foreach (ulong word in _code)
            if (!blockerSet.Contains(word))
                result.Add(word);
        result.AddRange(assignment.Select(option => option.Lifted));
        result.Add(candidate.Active);
        VerificationResult verification = Verifier.Verify(result, _n, _d, _w);
        return verification.Valid
            ? new SearchResult(
                result,
                candidate.Active,
                candidate.Blockers,
                assignment.Select(option => option.Deletion).ToArray(),
                assignment.Select(option => option.Addition).ToArray())
            : null;
    }

    private static ulong LowestBit(ulong value) => value & (0UL - value);

    private static int Binomial(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        k = Math.Min(k, n - k);
        int result = 1;
        for (int i = 1; i <= k; i++)
            result = checked(result * (n - k + i) / i);
        return result;
    }

    private static IEnumerable<ulong> EnumerateWeightWords(int n, int w)
    {
        ulong combination = (1UL << w) - 1;
        ulong limit = n == 64 ? 0 : 1UL << n;
        while (n == 64 || combination < limit)
        {
            yield return combination;
            ulong smallest = LowestBit(combination);
            ulong ripple = combination + smallest;
            if (ripple == 0)
                yield break;
            combination = (((ripple ^ combination) >> 2) / smallest) | ripple;
        }
    }
}

internal sealed record V2Candidate(ulong Active, ulong[] Blockers, RepairOption[][] Domains);
internal sealed record RepairOption(ulong Deletion, ulong Addition, ulong Lifted);

internal sealed record Candidate(ulong Active, ulong[] Blockers, ulong[] Domains, PairConstraint[] Constraints);
internal sealed record PairConstraint(int Left, int Right, ulong Intersection);
internal sealed record SearchResult(
    IReadOnlyList<ulong> Code,
    ulong ActiveWord,
    IReadOnlyList<ulong> Blockers,
    ulong[] DeletionBits,
    ulong[] AdditionBits);

internal sealed record VerificationResult(bool Valid, string Message, int MaxIntersection);

internal static class Verifier
{
    public static VerificationResult Verify(IReadOnlyList<ulong> words, int n, int d, int w)
    {
        ulong allowed = n == 64 ? ulong.MaxValue : (1UL << n) - 1;
        int lambda = w - d / 2;
        var seen = new HashSet<ulong>();
        int maximum = 0;
        for (int i = 0; i < words.Count; i++)
        {
            ulong word = words[i];
            if ((word & ~allowed) != 0)
                return new(false, $"word {i + 1} contains a coordinate above n={n}", maximum);
            int weight = BitOperations.PopCount(word);
            if (weight != w)
                return new(false, $"word {i + 1} has weight {weight}, expected {w}", maximum);
            if (!seen.Add(word))
                return new(false, $"word {i + 1} is a duplicate", maximum);
            for (int j = 0; j < i; j++)
            {
                int intersection = BitOperations.PopCount(word & words[j]);
                maximum = Math.Max(maximum, intersection);
                if (intersection > lambda)
                    return new(false,
                        $"words {j + 1} and {i + 1} intersect in {intersection} points (limit {lambda})",
                        maximum);
            }
        }
        return new(true, "ok", maximum);
    }
}

internal sealed record Code(IReadOnlyList<ulong> Words, int InferredLength, int InferredWeight);

internal static partial class CodeFile
{
    [GeneratedRegex(@"\d+")]
    private static partial Regex IntegerRegex();

    [GeneratedRegex(@"\(([^)]+)\)")]
    private static partial Regex CycleRegex();

    public static Code Read(string path, bool zeroBased, int? declaredN)
    {
        if (!File.Exists(path))
            throw new UserError($"Input file not found: {path}");
        string[] allLines = File.ReadAllLines(path);
        string? firstContent = allLines
            .Select(line => line.Trim())
            .FirstOrDefault(line => line.Length != 0 && !line.StartsWith('#'));
        if (firstContent?.Equals("$EXEC orbit", StringComparison.OrdinalIgnoreCase) == true)
            return ReadOrbitFile(allLines, declaredN);
        if (firstContent?.Equals("$BASE=16", StringComparison.OrdinalIgnoreCase) == true)
            return ReadHexFile(allLines, declaredN);

        var words = new List<ulong>();
        int inferredLength = 0;
        int inferredWeight = -1;
        int lineNumber = 0;
        foreach (string rawLine in File.ReadLines(path))
        {
            lineNumber++;
            string line = rawLine.Split('#', 2)[0].Trim();
            if (line.Length == 0)
                continue;
            ulong word;
            int length;
            if (line.All(character => character is '0' or '1') && line.Length > 1)
            {
                if (line.Length > 64)
                    throw new UserError($"Line {lineNumber}: bitstring exceeds 64 coordinates.");
                word = 0;
                for (int i = 0; i < line.Length; i++)
                    if (line[i] == '1')
                        word |= Program.Bit(i);
                length = line.Length;
            }
            else
            {
                MatchCollection matches = IntegerRegex().Matches(line);
                if (matches.Count == 0)
                    throw new UserError($"Line {lineNumber}: no coordinates found.");
                word = 0;
                int maximum = -1;
                foreach (Match match in matches)
                {
                    int parsed = int.Parse(match.Value, CultureInfo.InvariantCulture);
                    int coordinate = zeroBased ? parsed : parsed - 1;
                    if (coordinate is < 0 or > 63)
                        throw new UserError($"Line {lineNumber}: coordinate {parsed} is outside the supported range.");
                    ulong bit = Program.Bit(coordinate);
                    if ((word & bit) != 0)
                        throw new UserError($"Line {lineNumber}: coordinate {parsed} occurs twice.");
                    word |= bit;
                    maximum = Math.Max(maximum, coordinate);
                }
                length = maximum + 1;
            }

            int weight = BitOperations.PopCount(word);
            if (inferredWeight < 0)
                inferredWeight = weight;
            else if (weight != inferredWeight)
                throw new UserError($"Line {lineNumber}: weight {weight} differs from earlier weight {inferredWeight}.");
            inferredLength = Math.Max(inferredLength, length);
            words.Add(word);
        }

        if (words.Count == 0)
            throw new UserError("Input file contains no codewords.");
        if (declaredN is not null && inferredLength > declaredN)
            throw new UserError($"Input uses coordinate {inferredLength}, above --n {declaredN}.");
        return new(words, inferredLength, inferredWeight);
    }

    private static Code ReadHexFile(string[] lines, int? declaredN)
    {
        var words = new List<ulong>();
        int inferredLength = 0;
        int inferredWeight = -1;
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
        {
            string line = lines[lineIndex].Split('#', 2)[0].Trim();
            if (line.Length == 0)
                continue;
            if (line.Length > 16 || !line.All(Uri.IsHexDigit))
                throw new UserError($"Hex codeword line {lineIndex + 1} is invalid.");
            ulong word = ulong.Parse(line, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
            int weight = BitOperations.PopCount(word);
            if (inferredWeight < 0) inferredWeight = weight;
            else if (weight != inferredWeight)
                throw new UserError($"Hex codeword line {lineIndex + 1} has weight {weight}, expected {inferredWeight}.");
            int length = word == 0 ? 0 : 64 - BitOperations.LeadingZeroCount(word);
            inferredLength = Math.Max(inferredLength, length);
            words.Add(word);
        }
        if (words.Count == 0)
            throw new UserError("Hex input file contains no codewords.");
        if (declaredN is not null && inferredLength > declaredN)
            throw new UserError($"Hex input uses coordinate {inferredLength}, above --n {declaredN}.");
        return new(words, inferredLength, inferredWeight);
    }

    private static Code ReadOrbitFile(string[] lines, int? declaredN)
    {
        int separator = Array.FindIndex(lines, line => line.Trim() == "..");
        if (separator < 0)
            throw new UserError("Orbit file is missing its '..' generator/seed separator.");
        string[] seedLines = lines[(separator + 1)..]
            .Select(line => line.Trim())
            .Where(line => line.Length != 0 && !line.StartsWith('#'))
            .ToArray();
        if (seedLines.Length == 0)
            throw new UserError("Orbit file contains no seed words.");
        int sourceLength = seedLines[0].Length;
        if (sourceLength is < 1 or > 64 || seedLines.Any(line => line.Length != sourceLength || !line.All(c => c is '0' or '1')))
            throw new UserError("Orbit seeds must be equal-length binary strings of at most 64 bits.");
        if (declaredN is not null && sourceLength > declaredN)
            throw new UserError($"Orbit source length {sourceLength} exceeds --n {declaredN}.");

        var generators = new List<int[]>();
        for (int lineIndex = 1; lineIndex < separator; lineIndex++)
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;
            int[] permutation = Enumerable.Range(0, sourceLength).ToArray();
            MatchCollection cycles = CycleRegex().Matches(line);
            if (cycles.Count == 0)
                throw new UserError($"Orbit generator line {lineIndex + 1} contains no cycles.");
            foreach (Match cycle in cycles)
            {
                int[] points = cycle.Groups[1].Value
                    .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                    .Select(text => int.Parse(text, CultureInfo.InvariantCulture))
                    .ToArray();
                if (points.Any(point => point < 0 || point >= sourceLength))
                    throw new UserError($"Orbit generator line {lineIndex + 1} contains an out-of-range point.");
                for (int i = 0; i < points.Length; i++)
                    permutation[points[i]] = points[(i + 1) % points.Length];
            }
            generators.Add(permutation);
        }

        var expanded = new HashSet<ulong>();
        int weight = -1;
        foreach (string seedText in seedLines)
        {
            ulong seed = ParseBitString(seedText);
            int seedWeight = BitOperations.PopCount(seed);
            if (weight < 0) weight = seedWeight;
            else if (seedWeight != weight)
                throw new UserError("Orbit seed weights are inconsistent.");
            var queue = new Queue<ulong>();
            if (expanded.Add(seed))
                queue.Enqueue(seed);
            else
                queue.Enqueue(seed); // Its orbit may still not have been expanded from this seed.
            var orbitSeen = new HashSet<ulong> { seed };
            while (queue.Count != 0)
            {
                ulong current = queue.Dequeue();
                foreach (int[] generator in generators)
                {
                    ulong image = ApplyPermutation(current, generator);
                    expanded.Add(image);
                    if (orbitSeen.Add(image))
                        queue.Enqueue(image);
                }
            }
        }
        return new(expanded.Order().ToArray(), sourceLength, weight);
    }

    private static ulong ParseBitString(string line)
    {
        ulong word = 0;
        for (int i = 0; i < line.Length; i++)
            if (line[i] == '1')
                word |= Program.Bit(line.Length - 1 - i);
        return word;
    }

    private static ulong ApplyPermutation(ulong word, int[] permutation)
    {
        ulong image = 0;
        for (int source = 0; source < permutation.Length; source++)
            if ((word & Program.Bit(source)) != 0)
                image |= Program.Bit(permutation[source]);
        return image;
    }

    public static void Write(string path, IReadOnlyList<ulong> words, int n)
    {
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        using var writer = new StreamWriter(full, false, new UTF8Encoding(false));
        writer.WriteLine($"# {words.Count} constant-weight codewords; generated by ATS v1");
        foreach (ulong word in words.Order())
            writer.WriteLine(Program.FormatWord(word, n));
    }
}

internal static class CertificateFile
{
    public static void Write(string path, SearchResult result, int n, int d, int w, Command command)
    {
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        using var writer = new StreamWriter(full, false, new UTF8Encoding(false));
        writer.WriteLine($"# {(command == Command.SearchV2 ? "ATS v2" : "ATS v1")} gain-one trade certificate");
        writer.WriteLine($"parameters: n={n}, d={d}, w={w}");
        writer.WriteLine($"result_size: {result.Code.Count}");
        writer.WriteLine($"active: {Program.FormatWord(result.ActiveWord, n)}");
        writer.WriteLine($"trade: {result.Blockers.Count}->{result.Blockers.Count + 1}");
        writer.WriteLine("repairs:");
        for (int i = 0; i < result.Blockers.Count; i++)
        {
            ulong lifted = (result.Blockers[i] & ~result.DeletionBits[i]) | result.AdditionBits[i];
            writer.WriteLine(
                $"  {Program.FormatWord(result.Blockers[i], n)} " +
                $"- {BitOperations.TrailingZeroCount(result.DeletionBits[i]) + 1} " +
                $"+ {BitOperations.TrailingZeroCount(result.AdditionBits[i]) + 1} " +
                $"= {Program.FormatWord(lifted, n)}");
        }
        writer.WriteLine("verification: distinct weight-w words; maximum pair intersection <= w-d/2");
    }
}

internal sealed class UserError(string message) : Exception(message);
