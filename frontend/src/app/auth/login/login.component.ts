import { Component, OnInit } from '@angular/core';
import { AppService } from "../../app.service";
import { Router } from "@angular/router";

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {
  public login = 'admin@mail.ru';
  public pas = 'qwerty';
  public req: any;
  constructor(private service: AppService, private router: Router) { }

  ngOnInit(): void {
  }
  auth() {
    this.service.auth(this.login, this.pas).subscribe(value => {
      console.log(value);
      this.req = value;
      if (this.req.items) {
        localStorage.setItem('user', JSON.stringify(this.req.items[0]));
        this.router.navigate(['list']);
      }


    })
  }
}
