use std::io::Write;
use std::{thread, time};
use rand::Rng;

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
    fn scale(self, rhs: f32) -> Self {
        Vec2{x: self.x * rhs, y: self.y * rhs}
    }
    fn norm2(self) -> f32 {
        return self.x * self.x + self.y * self.y;
    }
}

struct Star {
    p: Vec2<f32>,
    v: Vec2<f32>,
    m: f32,
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
    let (width, _) = frame.resolution;
    let bpp = 4;
    let stride = width * bpp;
    let index = y * stride + x * bpp;
    frame.pixels[index + 0] = color.r;
    frame.pixels[index + 1] = color.g;
    frame.pixels[index + 2] = color.b;
    frame.pixels[index + 3] = color.a;
}

fn inside(p: Vec2<f32>, resolution: Resolution) -> bool {
    let (width, height) = resolution;
    p.x >= 0.0 && p.x < width as f32 && p.y >= 0.0 && p.y < height as f32
}

fn draw(frame: &mut Frame, stars: &Vec<Star>, zoom: &Zoom) {
    for star in stars {
        let screen = zoom.to_screen(star.p);
        if inside(screen, frame.resolution) {
            draw_pixel(frame, screen.x as usize, screen.y as usize, WHITE);
        }
    }
}

fn gravity_forces(stars: &Vec<Star>) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = stars.iter().map(|_| Vec2::zero()).collect();
    for i in 0..stars.len()-1 {
        for j in i+1..stars.len()  {
            let si = &stars[i];
            let sj = &stars[j];
            let delta = si.p.sub(sj.p);
            let r2 = delta.norm2();
            let f = G * si.m * sj.m / r2;  // gravity force
            let r = r2.sqrt();
            forces[i] = forces[i].add(delta.scale(-f / r));
            forces[j] = forces[j].add(delta.scale(f / r));
        }
    }
    forces
}

const G: f32 = 0.001;
fn step(stars: &mut Vec<Star>, dt: f32) {
    let forces = gravity_forces(stars);
    for i in 0..stars.len() {
        let force = forces[i];
        let star = &mut stars[i];
        let acceleration = force.scale(1.0 / star.m);
        star.p = star.p.add(star.v.scale(dt));
        star.v = star.v.add(acceleration.scale(dt));
    }
}

const FPS: f32 = 30.0;
fn main() -> std::result::Result<(), std::io::Error> {
    let mut frame = Frame::new((506, 253));
    let zoom = Zoom{center: Vec2::zero(), scale: 100.0, resolution: frame.resolution};
    let mut stars: Vec<Star> = vec!();
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let p = Vec2::make(
            rng.gen::<f32>() - 0.5,
            rng.gen::<f32>() - 0.5,
        );
        stars.push(Star{p: p, v: Vec2::zero(), m: 1.0});
    }

    let dt = 1.0 / FPS as f32;
    //loop {
    for _ in 0..100 {
        step(&mut stars, dt);
        draw(&mut frame, &stars, &zoom);
        std::io::stdout().write_all(&(frame.pixels)).unwrap();
        thread::sleep(time::Duration::from_secs_f32(dt));
    }
    Ok(())
}
