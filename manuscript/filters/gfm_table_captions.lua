-- gfm_table_captions.lua
--
-- Pandoc Lua filter that re-renders any Table AST node carrying a
-- non-empty caption as a raw HTML <table><caption>...</caption>...</table>
-- block. GFM's pipe-table syntax has no caption form, so Pandoc would
-- otherwise emit the caption as a bare paragraph after the table, losing
-- the table-caption semantics. Wired into the manuscript_jekyll rule only;
-- the LaTeX/PDF writer already handles Table captions natively.
--
-- This filter MUST run after --citeproc in the Jekyll pipeline because
-- table cells may contain Cite nodes (e.g. the studies table); resolving
-- citations first means pandoc.write here serializes the already-rendered
-- inline references rather than placeholder citation keys.

local function caption_is_empty(caption)
  if caption == nil then return true end
  local long = caption.long
  if long == nil or #long == 0 then return true end
  for _, block in ipairs(long) do
    if block.content and #block.content > 0 then return false end
  end
  return true
end

function Table(elem)
  if caption_is_empty(elem.caption) then return nil end
  local html = pandoc.write(pandoc.Pandoc({ elem }), 'html')
  return pandoc.RawBlock('html', html)
end
