#let SPC = 1cm

#set page(paper: "us-letter", margin: SPC, flipped: true)
#set block(spacing: 0pt)
#set text(font: "Adwaita Mono", size: 24pt)

#let alignments = (
  top + left,
  top + right,
  bottom + left,
  bottom + right,
)

#let metadata = (
  graphs: (
    num: 4,
    offset: 0,
    name: "graph",
    flipped: true,
  ),
  equations: (
    num: 4,
    offset: 200,
    name: "eqn",
    flipped: false,
  ),
  add: (
    num: 4,
    offset: 400,
    name: "add",
    flipped: false,
  ),
  range: (
    num: 4,
    offset: 600,
    name: "range",
    flipped: false,
  ),
  gemini: (
    num: 4,
    offset: 800,
    name: "confirm",
    flipped: false,
  ),
)

#let left-pad(num, length: 3) = {
  if num > 999 {
    return "error"
  }
  let stringified = str(num)
  return "0" * (length - stringified.len()) + stringified
}

#let single(data, num) = {
    for i in range(4) {
      place(alignments.at(calc.rem(i, 4)))[
        #image("markers/marker-" + left-pad(data.offset + (num * 4) + i) + ".png", width: 3cm)
      ]
    }
    place(bottom + center)[#upper(data.name) #num]
}

#let generate(data) = {
  set page(paper: "us-letter", margin: SPC, flipped: data.flipped)
  for num in range(data.num) {
    if data.flipped {
      single(data, num)
      if num != data.num - 1 {
        pagebreak()
      }
    } else {
      block(height: (100% - SPC) / 2, width: 100%, single(data, num))
      if calc.rem(num, 2) == 0 {
        v(SPC)
      }
    }
  }
}

#for item in metadata.values() {
  generate(item)
}
