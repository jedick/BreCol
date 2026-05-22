-- number_figures.lua
--
-- Pandoc Lua filter that prepends "Figure N: " to each figure caption in
-- document order. Intended for the Jekyll/GFM target only; the LaTeX/PDF
-- output relies on LaTeX's built-in figure counter and float labelling, so
-- this filter is deliberately not wired into the PDF rule in the Makefile.
--
-- Pandoc 3.x emits a Figure AST node for `![alt](path)` paragraphs (with
-- implicit_figures); the caption text lives in `caption.long` as a list of
-- Blocks. We prepend the numeric prefix to the inlines of the first such
-- block so the result reads e.g. "Figure 1: Classification pipelines."

local figure_counter = 0

local function prefix_inlines(num)
  return {
    pandoc.Str('Figure ' .. tostring(num) .. ':'),
    pandoc.Space(),
  }
end

function Figure(elem)
  figure_counter = figure_counter + 1
  local prefix = prefix_inlines(figure_counter)
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
