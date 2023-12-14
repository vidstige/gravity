#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::f32::consts::TAU;
use rand::prelude::*;

use eframe::{egui::{self, Ui, Sense, Id}, epaint::{Color32, Pos2, Vec2, Stroke, Rect}};
use gravity::Simulation;
use gravity;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Gravity App",
        options,
        Box::new(|_| { Box::<GravityApp>::default() }),
    )
}

type StarSet = Vec<usize>;

#[derive(PartialEq)]
enum Mode {
    Add,
    Remove,
    Force,
}

#[derive(PartialEq)]
enum AddMode {
    Single,
    Uniform,
    Spiral,
}

// Transforms world cordinates to screen cordinates (and back)
struct FromWorld {
    offset: Vec2,
    scale: f32,
}

impl FromWorld {
    fn transform(&self, position: &gravity::Vec2<f32>) -> Pos2 {
        Pos2::new(self.scale * position.x, self.scale * position.y) + self.offset
    }
    fn inverse(&self, position: Pos2) -> gravity::Vec2<f32> {
        let p = position - self.offset;
        gravity::Vec2 { x: p.x / self.scale, y: p.y / self.scale }
    }
    fn inversef(&self, f: f32) -> f32 {
        f / self.scale
    }

    fn pan(&mut self, drag_delta: Vec2) {
        self.offset += drag_delta;
    }

    fn zoom(&mut self, zoom_delta: f32, hover_pos: Pos2) {
        let old_scale = self.scale;
        self.scale *= zoom_delta.sqrt();
        self.offset = hover_pos - self.scale / old_scale * (hover_pos - self.offset);
    }
}

struct Play {
    on: bool,
    speed: f32,
    steps: usize,
}
impl Play {
    fn new() -> Play {
        Play { on: false, speed: 1.0, steps: 1 }
    }
}

struct GravityApp {
    rng: rand::rngs::ThreadRng,
    simulation: Simulation,
    // display parameters
    from_world: FromWorld,
    // play parameters
    play: Play,
    coefficients: Vec<(Vec<(f32, f32)>, String)>,
    // mode
    mode: Mode,
    radius: f32,
    // create mode parameters
    orbital_velocity: bool,
    add_mode: AddMode,
    spiral_arms: usize,
    add_speed: usize,
    mass: f32,
}

// galaxies code
fn random_point_in_circle(rng: &mut ThreadRng, radius: f32) -> Pos2 {
    let r = radius * rng.gen::<f32>().sqrt();
    let theta = rng.gen::<f32>() * TAU;
    Pos2::new(r * theta.cos(), r * theta.sin())
}

fn spiral_galaxy(rng: &mut ThreadRng, radius: f32, arms: f32) -> Pos2 {
    let inner = 0.2 * radius;
    let t = rng.gen::<f32>();
    let a = arms * t * TAU;
    let r = inner + t * (radius - inner);
    Pos2::new(r * a.cos(), r * a.sin())
}

impl Default for GravityApp {
    fn default() -> Self {
        Self {
            rng: rand::thread_rng(),
            simulation: Simulation::new(),
            from_world: FromWorld { offset: Vec2::ZERO, scale: 10.0 },
            play: Play::new(),
            coefficients: vec![
                (gravity::euler(), "Euler".to_owned()),
                (gravity::leap2(), "Leap 2".to_owned()),
                (gravity::ruth3(), "Ruth 3".to_owned()),
                (gravity::ruth4(), "Ruth 4".to_owned()),
            ],
            mode: Mode::Add,
            radius: 64.0,
            orbital_velocity: true,
            add_mode: AddMode::Uniform,
            spiral_arms: 5,
            add_speed: 1,
            mass: 1.0,
        }
    }
}
impl GravityApp {
    fn select(&self, pointer: Pos2) -> StarSet {
        let center = self.from_world.inverse(pointer);
        let r = self.from_world.inversef(self.radius);
        let r2 = r * r;
        self.simulation.state.positions
            .iter()
            .enumerate()
            .filter(|(_, &p)| p.sub(&center).norm2() < r2)
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    }
    fn sample(&mut self) -> Pos2 {
        match self.add_mode {
            AddMode::Single => Pos2::ZERO,
            AddMode::Uniform => random_point_in_circle(&mut self.rng, self.radius),
            AddMode::Spiral => spiral_galaxy(&mut self.rng, self.radius, self.spiral_arms as f32),
        }
    }
    fn add_star(&mut self, pointer: Pos2) {
        let relative_position = self.sample();
        let position = pointer + relative_position.to_vec2();
        let p = self.from_world.inverse(position);
        let velocity = if self.orbital_velocity {
            let indices = self.select(pointer);
            gravity::orbital_velocity(&self.simulation.center_of_mass(&indices), &p, self.simulation.g)
        } else {
            gravity::Vec2::zero()
        };
        self.simulation.add(&p, &velocity, self.mass);
    }
}

fn draw_grid(ui: &mut Ui, from_world: &FromWorld, target_spacing: Vec2) {
    let size = ui.available_size();

    let spacing_scale = from_world.scale / (2.0 as f32).powi(from_world.scale.log2() as i32);
    let spacing = target_spacing * spacing_scale;

    let color = Color32::from_rgb(0x44, 0x44, 0x44);
    let stroke = Stroke { width: 1.0, color};
    let painter = ui.painter();

    let o = Vec2::new(from_world.offset.x % spacing.x, from_world.offset.y % spacing.y);

    let grid_size = size / spacing;
    let (grid_width, grid_height) = (grid_size.x as usize + 2, grid_size.y as usize + 2);
    
    for gx in 0..grid_width {
        let x = (gx as f32 - 1.0) * spacing.x + o.x;
        painter.vline(x, 0.0..=size.y, stroke);
    }
    for gy in 0..grid_height {
        let y = (gy as f32 - 1.0) * spacing.y + o.y;
        painter.hline(0.0..=size.x, y, stroke);
    }
}

fn draw_stars(ui: &mut Ui, simulation: &Simulation, from_world: &FromWorld) {
    let painter = ui.painter();
    let color = Color32::from_rgba_premultiplied(00, 0x8b, 0x8b, 0x40);
    for position in simulation.state.positions.iter() {
        painter.circle_filled(from_world.transform(position), 3.0, color);
    }
}

impl eframe::App for GravityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        let spacing = Vec2::new(16.0, 16.0);
        egui::SidePanel::right("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.play.on, "play");
                ui.add(egui::Slider::new(&mut self.play.speed, -1.0..=10.0).text("speed"));
                ui.add(egui::Slider::new(&mut self.play.steps, 1..=10).text("sub-stepping"));

                ui.add(egui::Slider::new(&mut self.simulation.g, 0.1..=10.0).text("G"));
                ui.add(egui::Slider::new(&mut self.simulation.softening, 0.0..=10.0).text("softening (m)"));

                ui.checkbox(&mut self.simulation.barnes_hut, "barnes hut");
                if self.simulation.barnes_hut {
                    ui.add(egui::Slider::new(&mut self.simulation.theta, 0.0..=100.0).text("theta"));
                }
                
                ui.label("Symplectic integrator");
                for (coefficients, name) in self.coefficients.iter() {
                    if ui.add(egui::RadioButton::new(&self.simulation.coefficients == coefficients, name)).clicked() {
                        self.simulation.coefficients = coefficients.to_owned();
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Add, "Add");
                if self.mode == Mode::Add {
                    ui.checkbox(&mut self.orbital_velocity, "orbital velocity");
                    ui.add(egui::Slider::new(&mut self.mass, 1.0..=10000.0)
                        .text("mass")
                        .suffix("kg")
                        .logarithmic(true)
                    );
                    ui.add(egui::Slider::new(&mut self.add_speed, 1..=100).text("speed"));
                    ui.radio_value(&mut self.add_mode, AddMode::Single, "Single");
                    ui.radio_value(&mut self.add_mode, AddMode::Uniform, "Uniform");
                    ui.radio_value(&mut self.add_mode, AddMode::Spiral, "Spiral");
                    if self.add_mode == AddMode::Spiral {
                        ui.add(egui::Slider::new(&mut self.spiral_arms, 1..=13));
                    }

                }
                ui.selectable_value(&mut self.mode, Mode::Force, "Force");
                ui.selectable_value(&mut self.mode, Mode::Remove, "Remove");
            });
        });

        let id = Id::new("gravity_view");
        egui::CentralPanel::default().show(ctx, |ui| {
            // compute time step
            let real = ui.input(|i| i.unstable_dt).min(1.0 / 30.0);
            let dt = real * self.play.speed ;

            let rect = Rect::from_min_size(Pos2::ZERO, ui.available_size());
            
            ui.input(|i| {
                // control zoom
                if let Some(hover_pos) = i.pointer.hover_pos() {
                    self.from_world.zoom(i.zoom_delta(), hover_pos);
                }
                // control circle with mouse
                if i.scroll_delta != Vec2::ZERO {
                    self.radius = (self.radius + 0.1 * i.scroll_delta.y).clamp(0.0, 1024.0);
                }
            });

            draw_grid(ui, &self.from_world, spacing);
            draw_stars(ui, &self.simulation, &self.from_world);

            // display energy
            ui.label(format!("E = {:.1}", self.simulation.energy()));

            let response = ui.interact(rect, id, Sense::hover());
            if let Some(hover_pos) = response.hover_pos() {
                let stroke = Stroke { width: 1.0, color: Color32::WHITE};
                ui.painter().circle_stroke(hover_pos, self.radius, stroke);
            }
            
            let response = ui.interact(rect, id, Sense::click_and_drag());
            if response.clicked_by(egui::PointerButton::Primary) {
                if self.mode == Mode::Add && self.add_mode == AddMode::Single {
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        self.add_star(pointer_pos);
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    match self.mode {
                        Mode::Add => {
                            if self.add_mode == AddMode::Single { return; }
                            for _ in 0..self.add_speed {
                                self.add_star(pointer_pos);
                            }
                        },
                        Mode::Remove => {
                            let mut indices = self.select(pointer_pos);
                            indices.reverse();
                            for index in indices {
                                self.simulation.remove(index);
                            }
                        },
                        Mode::Force => {
                            let indices = self.select(pointer_pos);
                            // compute acceleration based on mouse movement

                            let a = 0.01 * response.drag_delta() / self.from_world.scale / dt;
                            println!("{:?}", a);
                            for index in indices {
                                self.simulation.state.velocities[index].x += a.x;
                                self.simulation.state.velocities[index].y += a.y;
                            }
                        },
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Secondary) {
                self.from_world.pan(response.drag_delta());
            }

            if self.play.on {
                // take a time step
                for _ in 0..self.play.steps {
                    self.simulation.step(dt / self.play.steps as f32);
                }

                // request a new timestep
                ui.ctx().request_repaint();
                //self.time += ;
            };
        });
    }
}