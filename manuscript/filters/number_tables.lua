-- number_tables.lua
--
-- Pandoc Lua filter that prepends "Table N: " to each table caption in
-- document order. Intended for the Jekyll/GFM target only; the LaTeX/PDF
-- output relies on LaTeX's built-in table counter and float labelling, so
-- this filter is deliberately not wired into the PDF rule in the Makefile.
--
-- Must run after inline_html_tables.lua (which attaches captions to the
-- linked-HTML tables from each link's title attribute) and before
-- gfm_table_captions.lua (which serializes Table nodes to raw HTML and
-- would otherwise bake in an un-numbered caption).

local table_counter = 0

local function prefix_inlines(num)
  return {
    pandoc.Str('Table ' .. tostring(num) .. ':'),
    pandoc.Space(),
  }
end

function Table(elem)
  table_counter = table_counter + 1
  local prefix = prefix_inlines(table_counter)
  local long = elem.caption and elem.caption.long

  if not long or #long == 0 then
    elem.caption = pandoc.Caption({ pandoc.Plain(prefix) })
    return elem
  end

  local first = long[1]
  if first and first.content then
    local new_content = pandoc.List({})
    for _, inline in ipairs(prefix) do new_content:insert(inline) end
    for _, inline in ipairs(first.content) do new_content:insert(inline) end
    first.content = new_content
  end
  return elem
end
