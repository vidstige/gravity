use std::io::Write;

type Resolution = (usize, usize);

struct Frame {
    resolution: Resolution,
    pixels: Vec<u8>,
}

impl Frame {
    fn new((width, height): Resolution) -> Self {
        Frame{resolution: (width, height), pixels: vec![0; width * height * 4]}
    }
}

#[derive(Clone, Copy)]
struct Vec2<T> {
    x: T,
    y: T,
}

impl Vec2<f32> {
    fn make(x: f32, y: f32) -> Self {
        Vec2{x: x, y: y}
    }
    fn zero() -> Self {
        Vec2{x: 0.0, y: 0.0}
    }
    fn add(self, rhs: Vec2<f32>) -> Self {
        Vec2{x: self.x + rhs.x, y: self.y + rhs.y}
    }
    fn sub(self, rhs: Vec2<f32>) -> Self {
        Vec2{x: self.x - rhs.x, y: self.y - rhs.y}
    }
}

struct Star {
    p: Vec2<f32>,
    v: Vec2<f32>,
}

struct Zoom {
    center: Vec2<f32>,
    scale: f32,
    resolution: Resolution,
}

impl Zoom {
    fn to_screen(&self, world: Vec2<f32>) -> Vec2<f32> {
        let (width, height) = self.resolution;
        Vec2::make(
            0.5 * width as f32 + self.scale * world.x,
            0.5 * height as f32 + self.scale * world.y,
        )
    }
}

struct Color {
    r: u8, g: u8, b: u8, a: u8,
}
const WHITE: Color = Color {r: 255, g: 255, b: 255, a: 128};

fn draw_pixel(frame: &mut Frame, x: usize, y: usize, color: Color) {
    let bpp = 4;
    let (width, _) = frame.resolution;
    let stride = width * bpp;
    let index = y * stride + x * bpp;
    frame.pixels[index + 0] = color.r;
    frame.pixels[index + 1] = color.g;
    frame.pixels[index + 2] = color.b;
    frame.pixels[index + 3] = color.a;
}

fn draw(frame: &mut Frame, stars: &Vec<Star>, zoom: &Zoom) {
    for star in stars {
        let screen = zoom.to_screen(star.p);
        draw_pixel(frame, screen.x as usize, screen.y as usize, WHITE);
    }
}

fn main() -> std::result::Result<(), std::io::Error> {
    let mut frame = Frame::new((506, 253));
    let zoom = Zoom{center: Vec2::zero(), scale: 1.0, resolution: frame.resolution};
    let mut stars: Vec<Star> = vec!();
    stars.push(Star{p: Vec2::zero(), v: Vec2::zero()});
    loop {
        draw(&mut frame, &stars, &zoom);
        std::io::stdout().write_all(&(frame.pixels)).unwrap();
    }
}
