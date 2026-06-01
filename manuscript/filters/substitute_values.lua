-- substitute_values.lua
--
-- Pandoc Lua filter that expands {key} placeholders in the manuscript body
-- using values from the document metadata. Combine with
-- `--metadata-file=values.yaml` (written by helpers/manuscript_values.py) to
-- substitute computed numbers and short fragments of inline Markdown into
-- the prose at build time.
--
-- Example:
--
--   manuscript.md:  ...the mean number of sequences used per sample is
--                   {hyenadna_sequences_per_sample_text}.
--   values.yaml:    hyenadna_sequences_per_sample_text: >-
--                     323 ± 112 (min 50 for [@YTK+26], max 540 for [@BVW+21])
--   rendered:       ...the mean number of sequences used per sample is 323 ±
--                   112 (min 50 for Yerlikaya et al. 2026, max 540 for ...).
--
-- The substituted text is re-parsed as Markdown so embedded citation keys
-- (`[@key]`), emphasis, etc. inside the value become proper AST nodes; this
-- matters because pandoc's Markdown reader runs before any --lua-filter, so
-- `[@key]` text injected later would otherwise reach --citeproc as a plain
-- Str element and never get resolved. Wired into both the PDF and Jekyll
-- pandoc invocations from the Makefile (before --citeproc in each).
--
-- Keys not present in the metadata are left as-is (so a missing value is
-- visible in the rendered document rather than silently dropped).

local function lookup_table(meta)
  local values = {}
  for key, val in pairs(meta) do
    values[key] = pandoc.utils.stringify(val)
  end
  return values
end

local function parse_inline_markdown(text)
  -- Returns an inlines list (possibly with Cite/Emph/etc.) for `text`. Falls
  -- back to a single Str element if the parse fails or yields no blocks.
  local ok, doc = pcall(pandoc.read, text, 'markdown')
  if not ok or not doc or #doc.blocks == 0 then
    return { pandoc.Str(text) }
  end
  return pandoc.utils.blocks_to_inlines(doc.blocks)
end

local function make_str_filter(values)
  return function(elem)
    local text = elem.text
    if not text or not text:find('{', 1, true) then
      return nil
    end
    local replaced = false
    local new_text = text:gsub('{([%w_]+)}', function(key)
      local v = values[key]
      if v == nil then
        return nil  -- leave the original "{key}" in place
      end
      replaced = true
      return v
    end)
    if not replaced then
      return nil
    end
    return parse_inline_markdown(new_text)
  end
end

function Pandoc(doc)
  local values = lookup_table(doc.meta)
  return doc:walk({ Str = make_str_filter(values) })
end
